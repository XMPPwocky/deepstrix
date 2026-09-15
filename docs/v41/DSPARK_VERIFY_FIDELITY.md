# The verify's fidelity, measured (2026-09-15)

`V41_DSPARK_XCHECK` compares row 0 of the batched verify against the decode
forward that immediately follows it: same token, same position, same prefix. Its
logits must match. This is the DIRECT correctness test — acceptance is a proxy,
and a bad one, because changing the split changes the generated text and
acceptance is content-dependent.

## Two confounds had to be removed before any number meant anything

1. **`rollback_kv` never restored the compressor store.** It carried only
   per-layer `(n_raw, raw_off)`, so a speculative batch's ~b/ratio compressor
   boundaries stayed in the compressed KV permanently, cumulatively, for decode
   to attend to. The accept path's partial rollback had the same hole, so
   REJECTED drafts polluted the compressed KV — a DSpark correctness bug, not
   just an instrument artifact. Fixed: `KvMark` now snapshots `n_comp`,
   `n_index_comp`, `state_kv`, `state_score` for the 4 owning layers (~32 KB).

2. **The probe perturbed decode through PAGER RESIDENCY.** A content fingerprint
   of every component a rollback must restore reported "restored all components"
   on every step — and the probe STILL changed what decode generated at
   temperature 0 (coherent English vs a hallucinated Chinese request). Under
   `V41_T2_CATCHALL=1` the box1/box2 split is decided by `pg.is_resident()`, so
   it is a function of request history; the probe pages experts and repartitions
   decode's own MoE, and f32 addition is not associative. `V41_T2_CATCHALL=2`
   makes the split a constant. `forward_layer.rs` documents this for decode; it
   had never been applied to a fidelity measurement.

## Result

Verify row 0 vs decode, mean logit cosine:

    B    original   +comp rollback   +deterministic split
    1      0.709        0.979              0.9988
    2      0.758        0.986              0.9989
    3      0.631        0.687              0.821
    6      0.603        0.710              0.867

**At B<=2 the batched verify reproduces decode essentially exactly.** The B>=3
gap is real and unexplained; it is the remaining fidelity defect.

## Refuted, each by measurement rather than by reading

| Hypothesis | Instrument | Result |
|---|---|---|
| Prefill MoE kernels differ from decode's | `V41_VERIFY_DECODE_MOE=1` | no change (0.673 vs 0.710 at B=6) |
| Prefill attention kernels differ | `V41_VERIFY_DECODE_ATTN=1` | small (0.771 vs 0.710) |
| `n_comp_after` wrong on the own-compressor branch | `V41_COMP_POSITIONAL=2` | ZERO disagreements with the positional formula |
| Two-lane split | `V41_SINGLE_LANE_AB=6` | 0.694 vs 0.679 — both arms broken, because the cliff is where a LANE first holds >1 row (lane_split = b/2 rounded up), not where a second lane appears |
| Rows 1.. leak into row 0 | poison rows 1.. (tokens AND carries, so the mHC pre-mix is covered) | row 0 does not move |
| Expert paging | same poison (poisoned rows reroute, changing the paged union) | row 0 does not move |
| Small-B catch-all, deterministic | `V41_SMALL_B_CATCHALL_DET=1` | fidelity 0.688 -> 0.519 for 1277 -> 1147 ms. Fifth variant of "move the verify's expert paging" to fail, and the first scored on fidelity instead of wall time. |

## Where the throughput actually stands

Warm, host tuning restored (it is NON-PERSISTENT and had been lost — restoring
it took baseline 4.26 -> 5.52 tok/s):

    baseline decode          181 ms/tok   = 5.5 tok/s
    DSpark accept, mode 1    890 ms/tok   = 1.1 tok/s   E 1.553
    DSpark accept, mode 2    649 ms/tok   = 1.5 tok/s   E 2.070
    verify(B=6) alone        1150-1280 ms

The deterministic split is worth real acceptance (E 1.553 -> 2.070), which
REFUTES the earlier "mode 2 tanks acceptance (E 1.07-1.39)" — that was measured
under the compressor bug.

**DSpark is currently a 4x net LOSS.** Break-even needs
`verify(B=6) < E x 181 ms` ~ 375 ms at E=2.07, against 1200 ms today: a 3.2x
verify cost reduction just to reach parity, and ~12x for 20 tok/s. E also swings
1.24-2.07 run to run on identical config, so acceptance is not yet a stable
quantity to optimise against.

## The verify CAN be made 2.9x faster — the blocker is correctness, not cost

Measured with `V41_T2_CATCHALL=2` (deterministic split, so the numbers mean
something) and the compressor rollback fixed:

| verify arm | prefill_misses | read_ms | verify total | cos |
|---|---|---|---|---|
| catch-all off | 298 | 724 | **1250 ms** | 0.814 |
| `_DET=1` (all picks to box 2) | **0** | **0** | **429 ms** | 0.521 |
| residency-based | 0 | 0 | 1370 ms | 0.512 |
| residency, decoder half only | 9 | 130 | 1423 ms | 0.569 |

So the offload DOES eliminate box 1's expert misses and takes the verify from
1250 ms to 429 ms — a 2.9x cut, and the miss elimination is real
(`prefill_misses` 298 -> 0). **Every variant loses fidelity the same way**
(cos ~0.52 against 0.81 with it off), so the remaining blocker is correctness,
not cost.

### What it is NOT

- **Not the static HELLO mask dropping picks.** `V41_MASK_DBG=1` reports zero
  masked-live picks: `remote_exclude()` is on by default, so `sel_for_remote` is
  built and the submit goes out UNMASKED.
- **Not a double-count from a stale remap.** The decode path needs
  `mark_remote_after_ensure` because a filtered-out id keeps `-(e)-1` ("ours at
  slot e"). The prefill path does not: `set_remote_exclusion` rewrites all
  `N_EXPERT` entries authoritatively from `slot_of` and marks anything not
  resident in the window as 0. Adding the decode guard here changed nothing
  (tested), and was reverted as redundant.

### The live hypothesis

At 429 ms with box 1 paging nothing, box 2 cannot have paged ~18 experts/layer
x 40 layers from its own disk (that would be seconds at 7.14 ms/miss). So box 2
is most likely returning partials only for the experts it already holds and
silently contributing nothing for the rest — the "computed by NOBODY" failure,
one level below the hub's own `verify_routing_exactly_once`, which validates the
hub's `owns_eff` and not box 2's behaviour. Next step is to instrument box 2's
side: count, per request, picks received vs experts actually computed.
