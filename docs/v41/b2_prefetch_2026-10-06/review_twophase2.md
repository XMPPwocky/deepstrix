# Verification: 479bdb6 (fixes to review_twophase.md)

## (a) Defaults = today's paths -- holds
- Whole-expert reads never stripe: `read_miss_into` sets `stripes = 1` for `ROLES_ALL` (6765) regardless of
  `role_stripes`; `read_expert_hf_layout_direct_routed` is now a thin call to `_striped(.., 1)` and
  `read_range_into_direct_striped` returns `read_range_into_direct_split` verbatim for `stripes <= 1` (93-95).
  The coalesced span path is untouched for three roles.
- The hook's DOWN branch is unreachable without partial slots: `early_page_plan` puts a pick in `down` only
  when `partial(e)` (302-318), and `is_partial_pool` is false for every slot unless a `ROLES_GATEUP` landing
  marked one, which needs `likely_gateup_only() && land_two_phase()` (forced conjunction, 279) or a GateUp-phase
  ensure (two-phase only). `words` is built exactly as before (non-resident, distinct, pick order).
- Per pass with the knobs off: `two_phase_stats()` early-returns constants (453-455), `ev_partial_fields` is
  arithmetic on them, `ev_gateup.record` is inside the `two_phase` branch, `prefetch_words_core` adds one
  `is_partial` HashMap lookup per word (tens per step), and `ensure` gains the `down_phase` bool only.
  `likely_gateup_only()` reads two atomics. Nothing else new runs.

## (b) `stripe_plan` / striped read -- holds, two notes
- Alignment: `span` is a multiple of 4096 (109), `per = ceil(blocks/n) * 4096`, `cut` is a 4096 multiple clamped
  to `[A, span - A]` (116-123), so every piece start/len is aligned, the straddling piece is split at the
  aligned cut, and the tail piece is `span - start` (a 4096 multiple). File offsets are `base + start` with
  `base = abs - pad` aligned; `dst` alignment is checked and `split_at_mut` at 4096 multiples keeps it.
- Coverage: `start` walks `0..span` contiguously; `pieces` are disjoint `split_at_mut` views of `dst[..span]`;
  each byte is read by exactly one thread. `need = (pad + len - start).min(plen)` makes only the last piece
  tolerate a short read (the file tail), as in the split read.
- Completion before repack: `thread::scope` joins every piece and the `?` over `results` runs before
  `Ok(Some(pad))`, so `read_miss_into` returns offsets only when all pieces of the role have landed; repack
  follows on the compute thread as before.
- NIT: chunked reads inside a piece advance `got` by `n`; a short (non-4096) `n` leaves the next pread offset
  unaligned -- the same behaviour as the existing split reader, so acceptable, but worth one comment.
- NIT: a 2-role read now spawns 2 x (stripes - 1) scoped threads per expert (~20-50 us each); fine at a few
  reads per step, but `role_stripes` 8 would be 14 spawns per expert -- cap the knob at 4 or pool the threads
  if the A/B ever raises it.

## (c) DOWN-only certain read -- holds
- Set accounting: `certain = true` so `pick_set` takes any general set (never the 2 x reserve drop rule, never
  the LIKELY pair); dropped only with no set at all, like today's early-page whole reads; released via
  `release_set` on every exit of the new landing branch (5435, 5447).
- Cannot land into a recycled slot: the landing (5416-5437) re-resolves `slot_of.get(&key)` at landing time
  and requires `partial[sl]`; a key evicted meanwhile has no slot (skipped), a key re-landed whole is not
  partial (skipped), and a key re-landed partial in ANOTHER slot receives its down bytes there -- correct,
  since bytes are keyed by expert, not slot. No generation counter is needed. A concurrent synchronous down read
  by the pass's Down phase cannot interleave with this landing: the Down phase runs no `admit_prefetched`
  (6323), so landings happen only in GateUp/Full where `must_wait` on a pending wanted key waits the down-only
  job out and `complete`s before the pass runs -- at worst a duplicate read, never a wrong slot.
- `nursery_protect`: `note_nursery_protect` keys on `in_nursery` (6043-6045), which includes partial nursery
  entries, so an arrived frame's partial entry is protected from recycling until served; the hook sets
  `shard.pinned = cur_pins` around `prefetch_down_words` as for whole reads.
- The dedup in `prefetch_words_core` (384): a down-only word for a non-partial key is skipped and a whole word
  for a partial key is skipped (the pass completes it) -- right both ways.

## (d) Down phase == one Full call -- holds in code, weakly tested
`down_phase` skips `admit_prefetched`, `note_serve`, `pg.requests` / hits accounting, `touch_hit`, and the
`B2_ENSURE` emit (6323, 6406-6418, 6642), returns at once when no wanted expert is partial (6512-6515), and
still accounts `read_ns` / `h2d_ns` / `two_phase_ns[1]` for the reads it does (correct: the read happened).
Finding (MINOR): the test (6689-6702) mirrors the gating by hand (`if phase != Down { note_serve }`) instead of
driving `ensure_layer_inner`, so it would pass even if the production gate regressed. A device-free drive is not
available (the read path needs the weights), so pin it another way: assert in the sim's odd seeds that
`serves[layer]` equals the number of passes (the sim calls `note_serve` once per pass already), and add a
daemon-side debug assertion at the end of a two-phase pass that `serves` advanced by exactly 1.

## Also verified
`down_exposed` = `ev_gateup` recorded after the gate/up half's q8k and queried after the Down phase (7594-7596):
complete -> the kernels finished before the read, exposed; `query` errors count as exposed (conservative).
`likely_gateup_only` forced off without `land_two_phase` (279) and tested. The paged predicate is a `match`
(6218-6222); the evtrace field order is one list on both emit paths.

VERDICT: APPROVE -- the knobs stay 0 in the bundle; the one change (tighten the (d) test as above) is not
blocking. The earlier minimum lists (loopback run incl. run 3, late = `moe_arrived`, SOFT clear at the switch,
re-price 3.4 on the striped per-role walls once measured) stand.
