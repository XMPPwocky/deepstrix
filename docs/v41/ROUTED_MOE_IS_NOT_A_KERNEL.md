# `igpu.routed_moe` is a 378-line host scope, not the iGPU MoE kernel
### found by the user in a perfetto trace, 2026-09-14

**Observation:** the `expert pager` host span appears to sit INSIDE the
`igpu.routed_moe` device span. It does. Literally.

`forward_layer.rs:2231` opens the stage and `:2609` closes it — 378 lines — and
the scope contains, on the HOST:

    2259  SEL_SYNC        de.compute.synchronize() + d_selected readback
    2446  submit_unmasked the network submit to box 2
    2483  pg.ensure       box 1's own expert paging
    2499  mark_remote_after_ensure

plus the actual iGPU MoE launches. So `igpu.routed_moe` measures the host's
router sync, box 1's paging and the remote submit **as well as** the kernel it is
named after.

`sel_sync_us` alone is **22,000-23,600 us/token** (~570 us/layer) — 9-10% of a
246 ms token spent waiting to learn six expert ids. That entire cost is reported
inside `igpu.routed_moe`.

## What this retracts

`WHY_THE_BIG_POOL_REGRESSED.md` attributed the 81.6 GB pool regression to
"`igpu.routed_moe` — box 1's own iGPU MoE kernel, +13.97 ms/token, +89%" and
priced box 1's iGPU at 140 us/expert from it. **That attribution is unsound.**
The stage grew by 13.97 ms, but the growth could be in the kernel, in
`pg.ensure`'s bookkeeping over 1,396 slots instead of 25, or in `sel_sync` —
the measurement cannot separate them. The regression itself is real and
reproduced (4.06-4.09 vs 3.26-3.60 tok/s, n=8); only the *cause* is now unknown
again, and the 140 us/expert figure derived from it must not be reused.

To attribute it properly, `ensure` and the submit need their own scopes outside
the MoE stage, or the kernel needs rocprofv3 (`reference_rocprofv3_kernel_trace`).

## Third time

This is the same failure as `feedback_price_the_part_not_the_stage`, which exists
in memory *because of an earlier instance*, and as the note in
`project_v41_prefill_32k_attn_share_2026-09-14` ("before quoting a profiler
number, verify WHICH code path emits it"). The rule needs strengthening: **before
attributing a stage delta to a kernel, read the scope's extent.** A `stage(...)`
guard in this codebase is an RAII scope that runs until end-of-block, and blocks
here are hundreds of lines long.

## The lever the observation exposes

The user's framing: *"as soon as the device finishes the router we should RACE to
tell box 2 what experts to start loading."* Today the order inside this scope is

    router done -> [~300 us] -> sel_sync -> pg.ensure -> submit to box 2

so box 2 learns its picks only after box 1 has finished its own paging. Sending
the ids the instant they are read back — before `pg.ensure` — lets box 2 start
its disk reads earlier by roughly the ensure window, every layer.

This is **exact prefetch, not predictive prefetch**. The predictive kind was
measured dead here (recall on misses 0.08-0.47,
`ROUTER_LOOKAHEAD_AND_PREFETCH.md`); this one has perfect information because the
picks already exist. Bounded by the hub's remaining per-layer work, so order
10-16 ms/token (5-6%), not the 6.6 ms of a whole miss.

## Two smaller answers from the same trace

**"Do we always emit a remote.expert span?"** Nearly. `submit_inner` returns
`Ok(None)` when no pick survives (`remote_experts.rs:2876`) and the hub's wait is
guarded by `if let Some(t) = ...take()` (`forward_layer.rs:2667`), so the no-op
path exists and costs nothing. But under T2 catch-all mode 2 the split is
`vec![true; N_EXPERT]` (`:2323`) — every routed pick is box 2's — so it fires on
every layer and the fast path never triggers. Saving it would require a split
that sometimes keeps a whole layer local, which today it never does.

**DPM.** Box 2 reports `power_dpm_force_performance_level = auto`. It does ~600 us
of work per layer and then idles ~5 ms, so clock ramp could be adding to every
layer's RTT. `deepstrix-expert-bench --gap-us N` exists to expose exactly this
wake-up cost. Untested: changing it needs root on box 2.
