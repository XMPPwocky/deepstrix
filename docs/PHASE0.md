# Phase 0 Hardware Viability — Measurement Report

Date: 2026-05-23. Machine: NixOS 26.05, ROCm 7.2.3, hipcc 22.0.0, rustc 1.95.0.
Hardware: AMD 9070 XT eGPU (gfx1201, RDNA 4) + AMD Strix Halo iGPU (gfx1151, RDNA 3.5).

## Gate summary

| Gate | Doc question | Outcome |
|---|---|---|
| Toolchain | Does `hipcc --genco` produce blobs `hipModuleLoadData` accepts? | **Yes**, CCOB bundles load directly, no unbundling needed |
| A — HSA override | Is `HSA_OVERRIDE_GFX_VERSION=11.5.1` compatible with dual-device? | **Override BREAKS the runtime.** Native works fine. Single-process, no override |
| B — gfx1100 perf | Is gfx1100 binary on Strix iGPU faster than gfx1151 native? | **No** (1.00× — bandwidth-bound, codegen doesn't matter). Stay native |
| C — peer access | Cross-device peer + events + coherency working? | Working with **load-bearing rule**: peer-copies MUST use src device's stream |
| D — MALL behavior | Does MALL hold residency? Do bypass hints work? | MALL **effective ~24 MB** (less than 32 spec), evicts under pollution, **non-temporal bypass works** |
| E — GEMV efficiency | Effective bandwidth vs theoretical? | 9070 XT: 94-97%. Strix iGPU: **107% of doc's claim** (doc was conservative) |

## Key measurements

### Bandwidth (Gate E, DRAM-bound shapes ≥136 MiB)

| Device | Theoretical (doc) | Measured p50 | Efficiency |
|---|---|---|---|
| 9070 XT | 644 GB/s | 604–622 GB/s | 94–97% |
| Strix iGPU | 215 GB/s | 229–231 GB/s | **107%** |

Strix exceeds the doc's number. LPDDR5X-8000 × 256-bit raw is 256 GB/s; we're hitting ~90% of that. The "215" figure in the doc was conservative; actual delivery is ~230.

### Cross-device sync (Gate C)

Steady-state, amortized over 200 events per batch, 20 batches:

| Direction | p50 sync cost |
|---|---|
| 0→1 (dGPU → iGPU) | **10.8 μs** |
| 1→0 (iGPU → dGPU) | **11.1 μs** |

Within the doc's 10–30 μs estimate. Asymmetry observed in one-shot RTT (27 vs 38 μs) was a host-polling artifact — vanishes when amortized.

**Per-token cross-device sync overhead: 4 transfers × 11 μs × 43 layers = 1.9 ms (~11% of 16.9 ms budget).** Workable.

### Peer-direct bandwidth (Gate C, p50)

| Size | 0→1 | 1→0 |
|---|---|---|
| 1 MiB | 6.7 GB/s | 6.9 GB/s |
| 16 MiB | 6.9 GB/s | 7.1 GB/s |
| 64 MiB | 7.0 GB/s | 7.2 GB/s |
| 256 MiB | 6.7 GB/s | 6.8 GB/s |

Matches doc's "7 GB/s after PCIe overhead" claim.

### Cache behavior (Gate D)

| | Strix MALL | 9070 XT IC |
|---|---|---|
| Spec | 32 MB | 64 MB |
| **Effective (full residency)** | **~24 MB** | **~64 MB** |
| DRAM rate | 172 GB/s | ~600 GB/s |
| Best cache rate | 530 GB/s (3.1× DRAM) | 1126 GB/s (1.9× DRAM) |
| Resilient to 32 MB pollution? | **No** — drops to DRAM rate | **No** — drops 998 → 448 |
| Non-temporal bypass effective? | **Yes** (511 → 207 GB/s) | Yes (1037 → 535 GB/s) |

## Design implications — sections of DESIGN.md that need updating

### §2 Hardware tables (soft numbers)
- Strix LPDDR5X effective bandwidth: replace 212-215 with **229–231 GB/s measured**
- 9070 XT effective bandwidth: replace theoretical 644 with **~605 GB/s sustained**
- Per-transfer setup latency: replace "1-5 μs" with **~5 μs submission + ~5 μs HSA signal propagation (one-way, steady-state)**
- Round-trip event sync: replace "10-30 μs estimate, unverified" with **~11 μs steady-state amortized, ~27 μs one-shot RTT**

### §2.4 Implications
- "Cross-device sync overhead may exceed assumed budget" — **DOES NOT exceed budget** in steady state (~11 μs amortized).

### §4.1 Device roles
- "Total: ~9-11 GB of 16 GB" on 9070 XT — verified, fine.
- "Total: ~80 GB of 96 GB GPU-addressable" on Strix — fine; HIP reports 134 GB but real allocatable is ~80 GB per the user's correction.

### §4.2 Per-layer dataflow
- Step 5 "40 μs bus + setup latency, unknown" — measured **~11 μs steady-state** (peer-direct on src_stream)
- Step 9 same
- The whole budget is more reachable than the doc feared.

### §4.3 Cache strategy — significant revision needed

Original plan: top-N = 8 experts × (W_up + W_gate) ≈ 35 MB ≈ MALL's 32 MB ("slightly overflows, reduce to N=7 = 30.5 MB if bad").

**Reality:** Strix MALL effective capacity is **24 MB, not 32 MB.** Even N=7 (30.5 MB) overflows. Real choices:
- **N=6** → 20.9 MB, comfortably fits in 24 MB MALL. Speculative-routing hit rate analysis needs to be redone for N=6 vs N=7/8.
- Accept partial residency at N=7 (~30.5 MB will give ~80% hit rate on bandwidth)
- Reduce per-expert weight size somehow (e.g. drop one of W_up or W_gate, but that's an arch change)

**Recommended:** N=6 in §4.4 with hit-rate retraining for the smaller pick set.

The non-temporal bypass for W_down is **viable** — confirmed working on Strix. Doc's plan to stream W_down with bypass hints holds up.

### §4.4 Speculative routing
- N=8 → N=6 per §4.3 revision above.
- Hit rate target ≥95% needs to be re-evaluated for N=6.
- The actual MALL prefetch window (~178 μs during attention) supports prefetching 24 MB at 230 GB/s → 104 μs. Plenty of margin even at full-bandwidth prefetch.

### §5.2 Synchronization model
- "**Phase 0 must verify** that `hipStreamWaitEvent` works correctly across the dGPU↔iGPU boundary with UMA on one side" — **Verified working.** Single-process design viable.
- **NEW LOAD-BEARING RULE**: peer-direct copies must be queued on the **source device's stream**, not destination's. Queuing on dst-stream silently returns zeros for dGPU→iGPU. See `memory/project-peer-copy-stream-rule.md`. Update the synchronization model description accordingly.

### §7.1 Toolchain — flip the override decision
- `HSA_OVERRIDE_GFX_VERSION=11.5.1` line should be **removed**. Phase 0 Gate A confirms this *breaks* dual-device (hipErrorLaunchFailure during `hipGetDeviceProperties`).
- gfx1100 binaries on Strix iGPU per §7.2 case 1: **no speedup** (Gate B). Drop the gfx1100 path; build only for gfx1201 + gfx1151 native.

### §7.2 Per-target kernel compilation
- Remove the "if HSA override works and gfx1100-binary-on-gfx1151 outperforms native" branch.
- Single canonical path: `--offload-arch=gfx1201,gfx1151`.

### §8 Phase 0 — mark complete with note that exit criteria are met.

### §9 Performance targets
The bandwidth efficiency findings (Strix 107%, 9070 XT 94-97%) combined with the steady-state sync cost (~1.9 ms/token) suggest the **55 tok/s target is reachable**, possibly **65 tok/s stretch with good overlap**. The bandwidth time alone is:
- 9070 XT: 122 MB attention + 27 MB shared = 149 MB × 43 / 605 GB/s = 10.6 ms/token
- Strix: 34 MB MoE × 43 / 230 GB/s = 6.4 ms/token (best case, all top-6 cached)
- LM head: 1 ms
- Cross-device sync: 1.9 ms

If attention and MoE overlap perfectly: max(10.6, 6.4) + 1 + 1.9 = **13.5 ms/token = 74 tok/s** (idealized).

If serial: 19.9 ms = 50 tok/s.

Realistic with partial overlap: 16–18 ms = **55–62 tok/s**. The doc's targets hold.

## Architectural decisions

1. **Single-process design.** No HSA override.
2. **Native arch kernels only.** gfx1201 + gfx1151. Drop the gfx1100-on-gfx1151 experiment.
3. **Peer-direct + events.** Use `hipMemcpyPeerAsync` queued on **source** device's stream.
4. **Top-N = 6** for routed-expert MALL prefetch (down from doc's N=8).
5. **Non-temporal W_down streaming** — `__builtin_nontemporal_load` confirmed bypasses MALL on Strix.
6. **Phase 0 exit criteria are met.** Phase 1 (single-device naive decode) is unblocked.
