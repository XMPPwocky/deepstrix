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
