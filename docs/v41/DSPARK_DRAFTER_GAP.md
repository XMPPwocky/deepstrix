# The drafter is the blocker, and it has two named deficits (2026-09-15)

After the verify's two bugs were fixed (`read_f32` missing stream sync; the
batched MoE never zeroing `partials`), the verify is cheap and can be faithful,
and DSpark's binding constraint moved to DRAFTER ACCEPTANCE.

Measured on the SAME transcript the CPU oracle was scored on
(`~/.cache/deepstrix/v41/agentic/prompt.txt`), greedy, K=5, E = 1 + mean accepted
prefix:

    source                 d1     d2     d3     d4     d5      E
    oracle base          0.843  0.730  0.674  0.607  0.562   4.382
    oracle noseed        0.764  0.618  0.461  0.348  0.213   3.281
    oracle nomarkov      0.640  0.461  0.371  0.326  0.292   2.382
    oracle legacy (bug)  0.438  0.258  0.135  0.067  0.056   1.820
    ENGINE               0.807  0.614  0.351  0.202  0.061   2.789
    ENGINE, no markov    0.649    -      -      -      -     1.965

The engine's FIRST draft is nearly the reference's (0.807 vs 0.843). It is DEPTH
that collapses: 0.061 vs 0.562 at d5, a 9x gap. It tracks `noseed` at d1-d2 and
falls BELOW it from d3.

## Deficit 1: the window ring is never seeded

`mtp.rs` says so outright:

    /// The reference indexes the window by `start_pos % window` and treats
    /// `min(window, start_pos + 1)` entries as valid, because its prefill seeds
    /// the window with the prompt's last `window` rows. Ours does not (yet) ...

So the engine IS the oracle's `noseed` configuration by construction. Worth
**+1.10 E** in the reference (3.281 -> 4.382), and both oracle seeding variants
(`base`, prefill-seeded; `incseed`, stepped one position at a time) reach the
same 4.382, so either implementation strategy works.

## Deficit 2: the markov head under-contributes

                with markov   without   contribution
    oracle         4.382       2.382       +2.00
    engine         2.789       1.965       +0.82

The head is active (disabling it costs the engine 0.82) but delivers 40% of what
the reference gets from it. Since the entry is parity-checked at cos 0.999958 and
the batched kernels at cos 1.000000, the fault is between those and the emitted
logits — the markov loop in `MtpExit`, or the rank-256 head's weights/bias.

## Not the cause (ruled out by measurement)

- **Alignment.** `dspark.shadow.align` probes: shift 0 gives mean_accepted 0.319,
  shift -1 gives 0.000, shift +1 gives 0.032. The engine's hidden/token pairing
  is right; the oracle's `ts±1` controls collapse to E~1.0 the same way.
- **The accept loop.** Shadow E (1.319) and accept-mode E (1.15-1.33) agree on
  the same content, so nothing is lost between drafting and accepting.
- **Verify fidelity.** A faithful verify (cos 0.999) gives LOWER E than a sloppy
  one (0.889) — a sloppy verify agrees with the drafter by accident. So fidelity
  is not what caps acceptance.

## What it is worth

At the reference's E=4.382 and a 150 ms B=6 verify, that is ~34 ms/token
(~29 tok/s) against decode's 67 ms/token. Both deficits must close AND box 2's
batched by-expert chain must be made to agree with its decode chain (currently
cos 0.889 vs 0.999), because the 150 ms figure is the batched chain's.

## CORRECTION: the drafter is NOT broken — it matches the reference

The numbers above (engine E 2.789 vs oracle 4.382) were measured on generations
too SHORT for the drafter's window ring to fill. The ring needs MTP_WINDOW = 128
writes; a 100-120 token run never reaches steady state.

Fresh server, ONE 500-token generation, same transcript:

    source                 d1     d2     d3     d4     d5      E
    ENGINE (500 tok)     0.798  0.629  0.439  0.309  0.180   3.077
    oracle noseed        0.764  0.618  0.461  0.348  0.213   3.281
    oracle base (seeded) 0.843  0.730  0.674  0.607  0.562   4.382

**The engine matches the oracle's `noseed` configuration**, which is exactly what
it is: `mtp.rs` says the prefill ring seeding is not implemented. The drafter is
correct. The whole gap to 4.382 is that one missing feature.

Wraparound past 128 is FINE (attention over keys is permutation-invariant given
correct per-key rope positions, and the ring overwrites oldest-first). Cross-
request pollution is real but small: `ring_writes` is never reset, so request 2
starts with request 1's keys, costing E 3.077 -> 2.972 (~3%).

## The real blocker: ACCEPT MODE CANNOT RUN LONG GENERATIONS

    dspark accept: partial rollback (1 of 5) refused: rollback_kv: layer 0
    wrapped since the mark (raw_off 0 < marked 2); the eviction-down copy moved
    the window, so the mark no longer addresses it

Once the context passes SWA_WINDOW, the eviction-down compaction moves the raw
window and every `KvMark` taken before it becomes unusable. The accept path
therefore dies on any generation longer than the window.

**This is why every accept-mode measurement in this session was wrong-headed.**
Accept mode is forced onto SHORT generations; short generations have a cold
drafter ring; so accept mode has only ever been measured where E is worst:

    accept mode (short, cold ring)   E 1.15 - 1.33
    shadow mode (long, filled ring)  E 3.077

The accept path has never been run in the regime where the drafter is good.

### The fix

The KV cache is deliberately sized `SWA_WINDOW + B_MAX` so a whole batch can be
appended WITHOUT evicting. The speculative append must therefore DEFER the
eviction-down until after the accept decision and rollback have settled, instead
of evicting mid-batch and destroying the information the mark needs. Eviction is
already a separate post-attention pass, so this is a matter of not running it
while a mark is outstanding.

At E=3.077 and the 149 ms batched verify that is ~48 ms/token (~20.7 tok/s)
against decode's 67 — but it also needs box 2's batched by-expert chain to agree
with its decode chain (cos 0.889 vs 0.999), since 149 ms is the batched chain's.
