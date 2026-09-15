# Why each attempt to move the verify's expert paging was wrong
### 2026-09-15, each cause identified, not just observed

The verify's cost is box 1 paging experts: at B=6, ~316 union-page requests,
~250 MISSES, ~700 ms of NVMe reads — 94% of a ~950 ms step — while box 2 sits
83% idle holding its own copy of every expert. Four attempts to move that work
failed. The causes:

## 1. All picks to box 2, `submit` (masked) — SILENT DROP

`submit_inner`'s mask is the **static HELLO bitmap** (`self.owns`), not the
hub's chosen split:

    if !mask || self.owns(layer, sel[i]) { sel_scratch[i] = sel[i]; ... }
    else { sel_scratch[i] = NO_PICK; ew_scratch[i] = 0.0; }

So every pick the hub reassigned that box 2 does not STATICALLY own became
`NO_PICK` and was dropped. Box 1 had already excluded them, so nobody computed
them. The comment on that very loop says it: "the hub marks a pick remote
because it does not hold it; if box 2 does not statically own it either, it is
silently [dropped]". I had read the doc comment on `submit`, not this.

## 2. Same, but `submit_unmasked` — DOUBLE COUNT

Removing the mask sends every pick, so box 2 also recomputed the experts box 1
kept. Two partials for the same expert are summed at `ffn_combine`. It also made
box 2 page the whole union itself: 1194 ms/token, acceptance 1.01.

## 3. Hub-side masking — WRONG SENTINEL CONSTANT (now fixed)

The right idea — mask box 2's pick list against the hub's own `owns_eff`, then
submit unmasked — but written with `SENTINEL_EXPERT` (`= N_EXPERT = 384`). That
is the LOCAL het-split convention, whose remap entry means "the other device
takes it". On the wire the empty-slot value is `NO_PICK` (`-1`). The daemon
rejected it outright:

    executor: layer 0 expert 384 (token 0) is not resident here

With `NO_PICK` and the matching `ew` slots zeroed, this is **acceptance-neutral**
(E 1.98-2.18 against a 2.0-2.3 baseline). Kept, but gated to fire only when the
hub actually reassigns something, so the default path stays byte-identical.

## 4. Residency catch-all (box 1 keeps residents, box 2 takes misses) — PARTLY CAPACITY

Still wrong even with the masking fixed: E 2.182 -> 1.124, consistently across
four runs. Restricting it to ENCODER layers recovers most of it (1.803), which
points at box 2's per-layer capacity — its daemon owns `116-383` on L0-L19 but
only `344-383` on L20-L39, with a `260/68` placement file, so a decoder layer
cannot seat ~30 extra experts and whatever does not seat is not computed. Not
fully explained: encoder-only still loses 2.18 -> 1.80.

## A caveat on the measurements

E has large run-to-run variance on an unchanged config — 1.46 / 1.66 / 1.98 /
2.18 / 2.26 — because T2 catch-all mode 1 makes output history-dependent, each
run generates different text, and acceptance is strongly content-dependent (the
oracle measures 2.2x between agentic and prose). The catch-all's deficit is real
(four runs at 1.10-1.23 vs a 2.0-2.3 band). The single-lane result (2.259 vs
1.735) is ONE pair and sits inside that band — it should be re-run several times
before being trusted either way.

## The rule this establishes

Wall time cannot tell a speedup from a wrong answer on this path, because the
cheapest way to go faster is to compute less. Every one of these returned
HTTP 200, passed `verify_routing_exactly_once` (which validates the hub's own
`owns_eff`, not what box 2 actually computed), and produced fluent text. Score
changes here with DSpark acceptance, and repeat the run enough to clear a
~0.5 E band.
