# Overnight plan, 2026-09-22 -> 23

Production at start: box 1 on `6c103f7`+ with `V41_MS_STAGGER=2` (ready-first), hot set 103,
merge=1 on box 2. Built but NOT deployed: stream-buffer fix (64 -> 16384 tokens, never
silently drop a token; dropped streams must log a distinct finish reason -- TODO).

## Window A -- short, can run any time the server is idle (~8 min)
1. Deploy the stream-buffer fix (first restart of the night).
2. Fidelity knob matrix on the contiguous verify path (`MS_STOP_AFTER=pf1`, dumps L0-2,
   `V41_T2_CATCHALL=2`): baseline (0.10786 nats) / `V41_VERIFY_DECODE_ATTN=1` /
   `+ V41_VERIFY_DECODE_MOE=1`. Decides whether DSpark verify is fixable by config.
   (09-15 said swapping attention alone cannot converge because q_normed had already
   diverged; tonight's dumps show q_normed IDENTICAL at layer 0, so re-test.)

## Main window (server down most of the night; box 2 up)
3. FIDELITY, production: arena rows are hard-wired to f16-score attention (they refuse
   f32 scores). First divergence vs decode is layer-0 `heads` (2e-4), amplified by Q8
   activation rounding (3e-3) and expert-routing flips to ~0.1 nats / 10-20% hidden
   state by L39. Fix: decode-exact (or f32-score) attention for arena rows. Gate: dumps
   match at L0, G5b back to its 0.02 bar, G5a-G5e. Price the throughput cost.
4. Driver throughput, ABBA, synthetic 8 rows, warmed: lockstep / staggered / ready-first.
   Make the winner the default.
5. Lookahead prefetch under ready-first at 1 row and 8 rows; row-aware default if it wins.
6. Single stream: perfetto trace, attribute the ~20 ms/token UNTIMED host time (+~10 ms
   timed readbacks) -- the 09-17 "pager host gap"; fix the cheap parts.
7. Single stream: box 2 hits-first at B=1 (`--decode-max-b 0`, i.e. batched path) --
   overlaps box 2's ~10.7 ms/token compute with its ~30 ms/token reads. NEEDS a box-2
   daemon restart by the user (plan: make decode_max_b a runtime knob so one restart
   covers the whole night).
8. DSpark feasibility: (a) acceptance re-measured if (2)/(3) fixed verify; (b) expert
   overlap across consecutive tokens = box-2 misses per ACCEPTED token in a verify batch.
   40 tok/s needs ~4.4 tokens/step at <=110 ms/step, i.e. box 2 busy <~100 ms per
   5-row verify (today ~195 ms at 5-6 rows). Design note for arena-native verify
   (multi-position rows, in-arena rollback, earlier-position lane must lead).
9. Admission waits: requests PARK when the KV arena lacks a contiguous run
   ("free 60384, largest run 38926") -- fragmentation. Investigate compaction-on-park.
10. Design only: keep finished generations retrievable after a client drop.

## Rules
- Check live streams before ANY restart (a 19:47 test restart killed a 45-min compaction).
- Fidelity runs use V41_T2_CATCHALL=2. Throughput A/Bs back-to-back, ABBA, per row count.
- Nothing ships without its gate. Restore production on the best VALIDATED config at the end.
