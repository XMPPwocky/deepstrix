# M40 — HIP graph parallelism investigation (postmortem)

**Status:** investigation suspended. Conclusions below; open questions for a future session.

## Goal

Make the `mhc_pre_attn_pair` / `mhc_pre_ffn_pair` blocks in `forward_pair_interleaved` faster by running the t0 / t1 independent kernels in parallel instead of serially.

Each block is 9 launches: 2× `rms_norm_no_weight` + 1× `f16_matvec_narrow_two_inputs` (2-wide, shared) + 2× `hc_sinkhorn_par` + 2× `hc_weighted_sum` + 2× `rms_norm_weighted`. The t0 and t1 instances are data-independent.

Single-stream capture (the existing baseline) forces a strictly linear FIFO chain — every node waits on the previous one even when there's no data dep. Theoretically a fork-join DAG should let t0/t1 pairs run on different CUs concurrently.

## Three approaches tried, all regressed perf

| approach | mhc_pre_attn_pair span (µs/call, p50) | pair bench (ms/pair, p50) | delta vs baseline |
|---|---|---|---|
| **Baseline: single-stream capture** | 111 | 65 | — |
| **Fused kernel** (1 kernel, 1 WG/token) | n/a | 91 | **+26 ms** |
| **Fork-join stream capture** (compute_t0/t1 + events) | 252 | 94 | **+29 ms** |
| **Explicit graph** (`hipGraphAddKernelNode` with explicit deps) | 216 | 86 | **+20 ms** |

## What we measured (rocprofv3 kernel trace, explicit-graph variant)

**Parallelism IS happening on device.** Out of 4128 consecutive `rms_norm_no_weight` pairs in the trace, 2743 had negative gaps (i.e. the second kernel started before the first ended — only possible on different CUs). Similar pattern for `hc_sinkhorn_par` and `rms_norm_weighted`.

**But the overlap is only ~70% of kernel duration.** Distribution of overlap magnitudes between consecutive same-kernel pairs:

```
rms_norm_no_weight overlap: p10=1.88 µs  p50=13.64 µs  p90=17.08 µs  max=22.2 µs
(kernel duration: p50=20.9 µs)
```

p50 overlap of 13.6 µs out of 20.9 µs duration → wall time per pair ≈ 27 µs instead of 41 µs serial — saves ~14 µs vs serial. Probably HIP scheduling latency dispatching the second WG to a different CU.

**Measured device wall of the full 9-kernel mhc_pre block (explicit graph): 84 µs p50.**

For comparison, captured graph runs the same kernels serially with sum ≈ 77 µs. So fork-join parallelism saves at most ~-7 µs of kernel wall — actually slightly *worse* because the parallel kernels run slightly slower than serial ones (resource scheduling overhead per kernel under contention).

## Where the explicit-graph regression actually came from

- Captured graph: span 111 µs = 77 µs kernel work + 34 µs launch overhead
- Explicit graph: span 216 µs = 84 µs kernel wall + **132 µs launch overhead**

The explicit graph adds ~100 µs of non-kernel overhead per graph launch versus the captured graph. **Whether this is per-node (~11 µs × 9 nodes) or per-launch (constant fixed cost) is unknown.**

Possible causes (untested):
- Captured graphs may go through a different/faster runtime dispatch path
- The `kernelParams` storage layout used by explicit construction may be less cache-friendly than what capture builds internally
- Per-node validation differences

## Cost of fork-join stream capture (separately)

Tried *capture-based* fork-join (multi-stream capture with `Event::record` + `Stream::wait_event` as fork/join markers): per-call regressed 111 → 252 µs.

That's ~140 µs of overhead — consistent with ~10 µs per event × 14 events used for 2 fork-join cycles (1 fork + 1 join requires record on origin, wait on two aux streams, record on each aux, wait on origin = 7 events; we used 2 cycles = 14 events).

So HIP capture-based fork-join pays ~10 µs per event during graph replay. **Capture-based fork-join is unusable for fine-grained ops** at this overhead level.

## Bigger lesson

Even with FREE parallelism (zero per-call overhead), the win for the mhc_pre block would be only ~7 µs/call × 86 calls/pair = ~0.6 ms/pair. The single-WG kernels in this block (rms_nw, sinkhorn, hcw, rms_w) are launch-latency-bound, not compute-bound; running two of them on different CUs doesn't speed them up much because each one is already small.

**For tiny launch-latency-bound ops on AMD HIP, fork-join parallelism gives marginal wins.** The high-leverage path for these kernels is **fewer launches** (fusion), not **parallel launches** (fork-join). The failed fusion attempt (M40-P8) had the wrong launch geometry (1 WG per token forced 24-row matvec to run serially), not the wrong concept.

## Open questions for a future session

1. **Is the 100 µs explicit-graph overhead per-node or per-launch?** Test: build minimal 2-node and 18-node explicit graphs, time them. If overhead is linear in nodes, per-node; if constant, per-launch. Outcome shapes whether fewer-bigger-nodes (per-node) or fewer-graphs (per-launch) is the right mitigation.
2. **Why is captured-graph dispatch ~3× cheaper than explicit?** Probably an internal fast path. Worth a deeper look at hipruntime / rocm internals — there might be flags or layout hints that unlock the same path for explicit construction.
3. **`hipGraphAddDependencies` + capture combo:** capture gives the fast dispatch path; surgically rewiring deps after capture might give us fork-join WITHOUT explicit-construction overhead. Untested. Requires `hipGraphRemoveDependencies` + `hipGraphAddDependencies` sys bindings (not yet added).
4. **What if all the small kernels were rolled into a single grid?** A custom kernel where each WG handles one (token, output-slot) pair could replace rms_nw + matvec + sinkhorn + hcw + rms_w with one launch — bypassing both the launch-overhead AND fork-join-overhead questions. Requires careful cross-WG sync (cooperative groups). High effort, possibly high payoff.

## Files left behind

- `crates/v4flash-kernels/tests/bench_mhc_pre_pair_microbench.rs` — per-call timing harness, useful to re-validate similar mysteries
- `crates/v4flash-hip/src/graph.rs` — `Graph::add_kernel_node` API added (works correctly, oracle confirmed bit-identical)
- `crates/v4flash-hip/src/sys.rs` — `hipGraphAddKernelNode` binding added
- `crates/v4flash-hip/src/module.rs` — `Function::raw_handle()` accessor

These all stay — they're useful primitives. The actual `mhc_pre_explicit_graph.rs` builder was deleted as dead code after revert.

## Verdict

For the mhc_pre block specifically: **don't pursue fork-join via either capture OR explicit construction.** The achievable win is small (~0.6 ms/pair if overhead were free) and the path to making overhead "free" is uncertain.

The pair-wall optimization budget is better spent on:
- **Cross-layer pipelining** to close the ~16 ms ffn_combine.wait gap (eliminates iGPU/dGPU serialization between layers, much bigger lever)
- **Per-layer mega-graph** that captures multiple consecutive stages together (recovers some launch overhead via fewer hipGraphLaunch calls)
