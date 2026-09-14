# DSpark: what exists, what is missing, and what it is worth
### grounded in measurements taken 2026-09-14

**DSpark is not implemented in the engine.** `grep` for dspark/speculative across
`het/` and `deepstrix-server` returns nothing; the E=4.13 figure every projection
rests on comes from `scripts/v41_oracle/dspark_accept.py`, a Python reference
model that never touches the engine. This plan says what it would take.

## What it is worth — two independent estimates that agree

**Estimate 1, from box 2's measured service model.** `srv_us = 105 + 20*B + 87*D`
per layer (fits 12 points to <3%), composed with the measured 71 ms zero-miss
floor and the real routing dedup (3.2x distinct experts at B=5):

    B=5 step 97.1 ms, E=4.13  ->  42.5 tok/s

**Estimate 2, from box 1's measured batched throughput.** Prefill wall time
against prompt length, max_tokens=1, warm, best of two:

    prompt tok    9    21    37    69   133
    wall s     1.32  1.61  1.82  2.24  2.66
    => slope 10.77 ms per token batched, intercept 1.23 s fixed per request

CED runs encoder layers 0-19 over all tokens plus a replay, so a full-depth
forward is ~2x that:

    B=4 step  86.2 ms, E=3.57  ->  41.4 tok/s
    B=5 step 107.7 ms, E=4.13  ->  38.3 tok/s
    B=6 step 129.2 ms, E=4.94  ->  38.2 tok/s

**42.5 and 38.3 tok/s from completely different measurement paths.** The headline
comparison is stark: a batched token costs ~10.8 ms where a sequential decode
token costs **246 ms** — 23x. That ratio, not the drafter, is the prize.

## What already exists

  * **A batched forward.** `forward_prefill` / `forward_prefill_pipelined` take
    `(tokens, pos0, last_only)` and return logits. A verify step is structurally
    a B-token chunk at the current position, which is exactly what prefill
    chunking already does on top of an existing KV.
  * **Verify routed to the right kernel.** SHIPPED today (`3ca1106`): the hub sets
    `REQ_FLAG_BATCHED` on multi-token submits, so a verify batch takes box 2's
    by-expert chain (+20 us/token) instead of its decode chain (+333 us/token).
    Measured 11-13% at B=2..4; B>=5 already escaped via `decode_max_b=4`.
  * **A batch-vs-sequential parity harness**, `forward_prompt_batch_matches_sequential.rs`
    — but for V4-Flash GGUF, and it tests batch-from-scratch, not continuation.
  * **KV designed for rollback.** `state.rs:279`: the decode path appends
    monotonically so "rejected appends just decrement, the evicted row is still
    in place."

## What is missing — three pieces

1. **Rollback.** Decrementing `n_raw` per layer is the easy half. The compressor
   and indexer-compressor carry streaming state that a rejected token also
   advances; `compressor_state_snapshot.hip` exists for snapshot/restore, so the
   primitive is available but must be wired and tested. **Testable without a
   drafter**: verify B known tokens, roll back, verify again, require bit-identity.

2. **The drafter.** `mtp.0/1/2` are in the checkpoint with trained `ffn_norm`
   gains (0.157/0.200/0.241 — the bug that made acceptance look like 0.44 was
   loading them as 1.0). Each stage is attention + MoE + router, i.e. a second
   model path. This is the bulk of the work.

3. **The accept/reject loop**, plus a decode loop that alternates draft and
   verify. Small once 1 and 2 exist.

## Recommended order

Build **rollback first**, because it is the only piece testable in isolation
(batch-verify / rollback / re-verify bit-identity on a warm KV) and it de-risks
the rest. Then a V4.1 continuation-parity test — `forward_prefill(B tokens at
pos p)` must equal B sequential `forward_token` calls — which turns the existing
batched path into a *validated* verify step. Only then port the drafter.

## The precondition that has not moved

Every number above assumes the zero-miss floor. Decode is 246 ms/token of which
~158 ms is expert misses. Speculation does not amortise misses: at B=5 the
distinct-expert count grows 3.2x for 4.13x the tokens, so misses/token falls only
23%. Today's measured best remains **~4.1-4.6 tok/s** (pool 52, n=4).
