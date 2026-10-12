# DSpark drafter latency (2026-10-12)

The drafter runs strictly between lone-stream DSpark steps: the next verify cannot start until the draft
exists. Measured on live traffic (54K lone cycles / 8 h, `ms.step` metadata): the gap between lone DSpark steps
equals `draft_ms` (~11.3 ms, flat in rows) = **14% of all lone DSpark time** (16-18% at 2-4 rows).

## Live split (before)

`V41_DSPARK_DRAFT_TIMING=1` + `V41_DSPARK_LAYER_TIMING=3` (mode 3 = device time from events, no serialisation),
4,199 drafts, p50:

| part | ms | device |
|---|---|---|
| total | 11.85 | |
| 3 layers: enqueue 0.66 + device 7.82 | 8.48 | iGPU |
| - MoE: gate+up 1.92, down 1.16, router 0.46, top-k 0.09 | 4.33 | |
| - attention: wo_a+wo_b 1.22, q_a 0.73, kv 0.31, q loop 0.29 | 2.61 | |
| - mHC mixing | 0.80 | |
| host hand-off (410 KB) | 0.08 | |
| exit: tied vocab head (703 MB Q8_0, ~1.1 ms at ~630 GB/s) + markov loop | 3.22 | dGPU |

Roofline: the big streams (wo_a/wo_b 197-205 GB/s, q 194, routed experts at ~12 distinct per layer) sit at the
iGPU's practical ceiling (~215 GB/s); ~1.8 ms/draft was latency-bound small kernels. Dense-on-dGPU (~3 ms/draft)
is blocked by dGPU VRAM (245 MiB free vs ~0.55 GiB needed).

## Changes (live knobs, default on; each gated bit-identical)

- `V41_MTP_MHC_FAST`: each mHC sub-block (RMS + 24-row mix matvec per row + Sinkhorn + carry + collapse + norm,
  ~10 launches) as one `MhcArena::launch_fast` (`MIX_NORMED`). iGPU 89 -> 42 us per sub-block, 6 per draft.
  `tests/mtp_mhc_fast_bitexact.rs` (both GPUs).
- `V41_MTP_ROUTER_BATCHED`: router as one `f16.matvec_batched` + one `router_topk.launch_batched` (was 5 + 5).
  iGPU 171 -> 30 us per layer. Same test file, near-tie seeds included.
- `V41_MTP_DEVICE_MARKOV`: the exit's markov loop with every position's token in device memory
  (`kernels/mtp_markov.hip`: row gather by device id; `mtp_draft_draw` = `draft_dist`'s draw in f64 over 64 lanes;
  a 64-WG argmax with `argmax_one`'s exact order) and the top-M by the indexer's exact threshold select
  (`IndexerTopkBitonic::launch_batched`, no bitmap, `done`); one sync per draft. The host still computes the
  verifier's q with `draft_dist` and re-draws to check (`MARKOV_DRAW_MISMATCH`). `tests/mtp_device_markov.rs`:
  drafts, q, confidence, biased logits identical (greedy + 3 sampling settings). The loop was DEVICE-bound, not
  round-trip-bound: argmax_one (1 WG over 129,280) 135-155 us -> 14 us, top-M 126-192 -> 83 us per position.

Note: Q8_0 device weights are REPACKED per row (all f16 scales, then all i8 quants); synthetic Q8_0 test data
must use that layout.

## Result (live, hub f9b9de1f from 2026-10-12 17:05 UTC)

Window gate: bit-identical to the deployed source (244 logits files, 132,993 picks; G5 drafts greedily, so the
sampled device draw is covered by its own test and the live mismatch counter: 0). Live p50: draft **11.85 ->
10.57 ms (-11%)**; mixing 0.84 -> 0.30, router + top-k 0.55 -> 0.20, exit 3.22 -> 2.76; `ms.step` `draft_ms`
11.80 -> 10.60 = lone DSpark cycles ~1.3-2% shorter.

Open: the exit's slow tail right after a restart (p90 3.95 in the first 15 min, falling over time; under
investigation); a multi-WG top-M select (83 -> ~20 us per position); the drafter's kv rows (60 launches, <= 0.2 ms).
