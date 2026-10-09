# Nursery draft for rev 3 (owner direction 10-06; to be merged once round-2 findings arrive)

Pool facts (remote_experts.rs): ShardPool 2240-2271 (owner_of per slot, slot_of, last_use with the
PREFILL_AGE class offset, remap_hosts[layer][e] = -(slot)-1 | 0, dirty per layer, held/floor per layer,
pins: PinBook 2726-2894, stage band boundary `stage` + StageCounters, me: ModeEvict 2381-2416 with the
restore list + restore_inflight; RESTORE_PUMP 4 / RESTORE_INFLIGHT 2). pick_victim 3205-3257 (range,
never want/extra/pinned, floors), pick_victim_any 3264-3285 (Band::Main|Stage), evict 3291-3303 (THE
choke point; pin violation if pinned), touch_hit 3307-3330, claim_miss 3347-3388, commit 3391,
land 3400-3408, unclaim 3412. Landing of background reads: admit_prefetched 4528-4617 (victim via
pick_victim_any, evict, repack_in_place, land / land_stamped). residency_words 4841-4858 =
remap_hosts != 0. PinBook::report 2858-2893: pinnable = slot < stage (a staging slot is landed but
never pinned) -- the precedent for a band that never pins. RESP flags: RESID 1<<15, PIN 1<<14; 1<<13
free. Knobs: `Knob::int/flag(...).alias("name")` in knobs mod 1948-1984 (file ~/expertd-knobs.txt, SIGUSR2).

## Nursery = a floating set of N slots (relabel, no copy)

- Per-slot flag `nursery: Vec<bool>` (not an address range): the N slots currently reserved. Main-band
  searches (pick_victim for claims/landings/restores) skip nursery slots; nursery landings pick only
  among nursery slots, oldest landing first (per-slot land tick), preferring entries whose target
  layer has since been served by >= `lanes` requests (`served_since_land`), i.e. unused.
- Hint landing (PfDone.likely): victim = nursery slot per the rule above; `evict` it (never pinned by
  construction: PinBook::report treats a nursery slot as non-pinnable like a staging slot, and grant()
  for a nursery expert is deferred); repack in place; `land` with nursery stamp (last_use = 0-class so
  it never competes); remap_hosts points at it, so the next ensure HITS it; `held[layer]` counts it
  (floors unaffected: nursery slots are excluded from floor protection like staging). Counter
  nursery_lands; if the evicted occupant was an unused hint: nursery_recycled.
- Promote-on-use: ensure's hit path (touch_hit) finds slot_of -> if nursery[slot]: clear the flag,
  stamp as a decode hit (tick + PREFILL_AGE), me_account as a demand claim would, pin eligibility
  via the request's normal pin_grant/pin_report (the slot is landed and now pinnable), counter
  nursery_hits; then refill the reserve: pick_victim_any(Main, never pinned, floors honoured) -> evict
  -> flag that slot nursery (free). That eviction is exactly the one the demand read would have
  caused; if no victim exists the nursery shrinks by one (count < N, refilled at the next free /
  eviction). No copy, no remap rewrite for the promoted expert; the victim's layer goes dirty as any
  eviction does.
- Why relabel not copy: a fixed band would need an 18.8 MB copy per promotion on the request path
  (iGPU copy ~0.1-0.2 ms) or an async second landing; relabel is O(1) and keeps the kernel reading
  the slot it already maps.
- Size: lanes x cap per lane-layer x lead layers = 2 x 8 x 1 = 16 at the cap (k1); 32 for k2.
  Knob `nursery` (V41_B2_NURSERY, default 0 = off, 16 suggested), carved from the pool at startup
  (4480 -> 4464 main + 16) or on top with the +270 restart. 16 slots = 300 MB.

## Reader class

PfJob.class: Certain > Likely > Spec. push: Likely behind certain, ahead of spec (1541-1552).
pop: a Likely job runs when no certain job waits; it uses any reader (certain reserve kept);
background_should_wait (1590-1598) also pauses speculative chunks while running_likely > 0; a Likely
read neither yields (4245-4262) nor chunks; staging sets: Likely words take sets down to the certain
reserve (not the 2x rule at 4342), so a restore burst cannot drop them; promote() of a Likely key by
the demand ensure works as today. Applied at FRAME ARRIVAL in the early-page hook (7196-7224: decode
nreq.prefetch when the frame carries REQ_FLAG_LIKELY 2048) -> +0.26 ms lead vs dequeue (7147).
b2_read src = 4 (likely). Falls back to the plain speculative path when the daemon does not echo the
bit (hub detects from the echoed flags).

## Residency map / mirror

residency_words masks nursery slots (bit = main-landed only) so the hub's `held` never includes a
nursery entry; PinBook::report never pins one. New RESP_FLAG_NURSERY 1<<13: 12 words appended after
the pin block when the request carried REQ_FLAG_LIKELY: bit e = expert e of `layer` sits in the
nursery. Hub `b2_mirror` keeps a NURSERY bitset per layer: used for (1) hint dedup across lanes and
steps (never re-hint a nursery entry or a pending one), (2) counters (`lh2_nursery_covered` = a
predicted miss covered by the nursery, i.e. an expected hit; `lh2_nursery_unused`), (3) NOT for
`held`: the cache prior stays hint-blind, so I2 ("hints never change routing") holds exactly even
with the prior on, and `n_pred_miss`/`sub_blocked` keep their meaning (the covered share is the new
counter). Argument against letting the prior see the nursery: its bias toward held experts would turn
a hint into a pick (feedback: hint -> prior -> pick -> promote), making wrong hints self-confirming;
the only upside (not swapping a rank-3 pick that is cheaply available) is rare at R <= 2. Knob
V41_B2_NURSERY_PRIOR (default 0) for a later golden-gated trial.

## Restore list / pins / admit gate interplay

Hints bypass admit_passes and the pin ledger entirely (no note_admits, no grant until promoted); the
TinyLFU watermark and released_unused are untouched by hints. Restores (delta restore, main-pool
landings over decode victims, RESTORE_INFLIGHT 2) keep their readers/sets/victims; nursery landings
never take main victims; the only main eviction a hint ever causes is at promotion (= the demand
read's). Restore pump throttle (free sets > half) sees fewer free sets while hints are in flight -- fine.

## Counters (behavior-free; b2_req fields + hub_step)

box 2: nursery_lands, nursery_hits (promotions), nursery_recycled (unused evicted by a later hint),
nursery_drops (no staging set), nursery_occupied, nursery_shrunk (promotion without a refill victim);
invariant per step: lands = hits + recycled + delta(occupied) (+ drops never land). hub: lh2_hints_sent,
lh2_nursery_covered, lh2_paged_hinted (tail recall), promotion rate hits/lands (precision; Step 0 says
0.71 R=1 / 0.87 any-pick R=2), churn = recycled/step (= wrong hints), main_evictions/step unchanged
vs nursery off minus promotions.

## Gates (daemon)

Unit: ring/recycle order, relabel promotion + refill victim honouring pins/floors, nursery never in
RESID words, never pinned (PinBook state stays NONE while in nursery; grant deferred), surprise check
unaffected (a nursery hit is a hit, PAGED bit clear), 6-seed randomized protocol sim extended with
Likely words. iGPU loopback (beside remote_experts_pin_loopback): hints on/off partials bit-identical,
promoted entries served from their slot, 0 pinned evictions, lands = hits + recycled + occupied delta.
Deploy: box 2 first (two-box order), hub falls back to (b) until RESP_FLAG_NURSERY is seen.

## Decision update

Nursery = (a)'s mechanism and the TARGET; hub-only (b) = interim/measurement path (slices A-C).
Price (price2.log, (a) at frame arrival): R=1 4.1 ms/step (5.1%), R=2 4.8 (6.0%), +k2 5.8 (7.6%);
with the nursery R=2 is the default (wrong hints are one slot of churn, not pins), vs (b) 2.3-3.0.
