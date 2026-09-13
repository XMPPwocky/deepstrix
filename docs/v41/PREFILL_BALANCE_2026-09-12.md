# Prefill device balance at 32K — MEASURED 2026-09-12 (and it inverts the June finding)

**The dGPU is the prefill ceiling, not the iGPU.** This kills the "stream experts to the dGPU
for its spare FLOPs" idea on V4-Flash, and removes most of the V4-Flash motivation for
super-chunk layer-major prefill. It does not invalidate the V4.1 plan, but it does invalidate a
premise that plan leans on (PLAN §5.3).

## Run
`bench_prefill_chunked`, T=32768, production quant (Vision-Exp UD-IQ3_XXS), `PIPELINE_LANES=2`,
`DGPU_HOT_EXPERTS=15`, `IGPU_DEDUP_HOT=1`, perfetto attached after warmup, chunk = 512 rows/lane.
Throughput **716 tok/s** — in line with production (735 @4K, 683 @96K), so the instrumentation
overhead is negligible and the trace is representative.

## Balance (64 chunks, 45.76 s wall)

| track | compute busy | % of span | idle |
|---|---|---|---|
| **dgpu.compute** | **39.30 s** | **85.9%** | 14.1% |
| igpu.compute | 28.46 s | 62.3% | 37.7% |
| dgpu.xfer | 3.45 s | 7.5% | — |
| igpu.xfer | 3.37 s | 7.4% | — |

The dGPU does **38% more device work** than the iGPU. `wait_event` idle is 0 on every track;
the dGPU's 14% idle is dominated by one transition, `dgpu.hot_moe_prefill → k.ffn_combine.vec_add`
(~1.4 ms avg, ~60% of all dGPU idle) — that is the cross-device join waiting on iGPU experts,
classified as a host gap because the join is not an event wait.

**The June 2026 note "iGPU is the prefill ceiling at PIPELINE_LANES=2" is STALE.** Since then the
iGPU gained kwide (+39%) and then the IQ2_S/IQ3 WMMA MoE path, while the dGPU picked up attention
work that grows with context. The ceiling moved.

## Where the dGPU's time goes (top stages, % of its 39.30 s)

| stage | total | calls | avg | share |
|---|---|---|---|---|
| `dgpu.kv_append_compressor_serial` | 11.07 s | 2752 | 4023 µs | **28.2%** |
| `dgpu.hot_moe_prefill` | 4.37 s | 2752 | 1587 µs | 11.1% |
| `k.attn.swa` | 2.07 s | 128 | 16203 µs | 5.3% |
| `k.attn.smwsum` | 1.62 s | 2624 | 616 µs | 4.1% |
| `k.mhc_pre_attn.f16_matvec` | 1.54 s | 2752 | 559 µs | 3.9% |
| `k.mhc_pre_ffn.f16_matvec` | 1.50 s | 2752 | 544 µs | 3.8% |
| `k.q_chain.rms_nw_heads` | 1.42 s | 2752 | 517 µs | 3.6% |
| `k.attn.score` | 1.40 s | 2624 | 534 µs | 3.6% |

`kv_append_compressor_serial` alone is **28% of the binding device**, and its variance is extreme
(p10 1.33 ms, p50 1.49, p90 7.77, p99 8.04) — consistent with the compressor firing on group
boundaries. This is the same "compressor f16 pair matvec (biggest flat kernel)" flagged as the
next lever on 2026-09-08 and never taken.

## Consequences

1. **Streaming experts to the dGPU in prefill is backwards on V4-Flash.** It would move work onto
   the device that is already 86% busy, from the one with 38% slack. Dead unless the dGPU is
   unloaded first.
2. **Super-chunk layer-major buys ~nothing on V4-Flash prefill.** Its mechanism is amortizing
   *iGPU* expert weight reads; the iGPU is not the bottleneck. The earlier "+33%" estimate assumed
   the June balance and is withdrawn. Layer-major would only start paying after the dGPU drops
   ~28% — i.e. after the compressor stage is fixed.
3. **The real V4-Flash prefill lever is `kv_append_compressor_serial`.** Cutting it in half is
   worth ~14% of the binding device, which is worth more than anything else on the list.
4. **PLAN §5.3 (stream experts to the dGPU during V4.1 prefill) rests on an unverified premise.**
   Its claim of "~50% dGPU spare after attention" is exactly the assumption that just failed here.
   V4.1 has a structural reason to differ — CED runs only the 20 encoder layers during prefill,
   halving the dGPU attention leg — but until that is measured the streaming idea should be
   treated as unproven, not planned-for.

## Caveat
One depth (32K), one quant, `last_only=true`. The balance is depth-dependent: attention and
indexer work grow with context, so the dGPU's share should be *lower* at 4K and *higher* at 192K.
Worth a second point at 4K before acting on item 3.
