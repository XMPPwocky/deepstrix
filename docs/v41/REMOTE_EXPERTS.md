# Remote expert executor — `deepstrix-expertd` (M8, 2026-09-13/14)

Expert-parallel topology from PLAN.md §4 / §7b: the hub (box 1: dGPU + iGPU) runs attention,
routers, shared/hot experts and its own iGPU expert share; box 2 (128 GB, iGPU only) holds a
large share of the routed experts resident in its iGPU pool and computes weighted MoE partial
sums for the hub over the USB4 link. This document is the protocol, the code layout, the
measurements, and exactly what is left to wire into `forward_layer` / `forward_prefill`.

Status: **built, loopback bit-identical, cross-box measured, clock-synced and traced** (numbers in
§5). Not yet called from the forward pass (§6 is the proposed call site). Nothing committed.

## 1. Code layout

| file | what |
|---|---|
| `crates/v4flash-kernels/src/het/remote_experts.rs` | everything shared: `proto` (frames), `Assignment` (which `(layer, expert)`s a daemon holds), `ExpertShard` (daemon-side resident pool), `MoeExecutor` (the iGPU MoE for one request), `serve`/`serve_connection` (daemon loop), `RemoteExpertClient` (hub side), `summarize` (timing percentiles) |
| `crates/deepstrix-expertd/src/main.rs` | `deepstrix-expertd` daemon binary (std::net only, hand-parsed args, no new dependencies) |
| `crates/deepstrix-expertd/src/bin/bench.rs` | `deepstrix-expert-bench`: hub-side client bench + cross-box bit-identity check |
| `crates/v4flash-kernels/tests/remote_experts_loopback.rs` | loopback test on one box: daemon thread with a 16-expert shard vs an INDEPENDENT dense-layout reference; timing |
| `crates/v4flash-kernels/src/het/perfetto.rs` | **additive only**: `TrackExporter` (open track set, device + host-time slices) alongside the existing `DeviceTimingExporter`, which is untouched |
| `crates/v4flash-kernels/src/het/mod.rs`, root `Cargo.toml` | one `pub mod` line, one workspace member line |

Untouched (other agents own them): `forward_prefill.rs`, `forward_layer.rs`, `expert_pager.rs`,
`weights.rs`. The executor calls the same public dispatch functions those files call
(`het::dispatch::moe_gate_up_batch_hetsplit`, `moe_down_batched_hetsplit`, `moe_gate_up_chunked`,
`MoeGroupBuilder::launch_hetsplit/launch_work_items`, `Mxfp4Matvec::launch_by_expert_kwide2`,
`Q2KAccumulateMatvec::launch_reduce_partials_hetsplit`, `Q8KQuantize::launch/launch_cast_f16`).

Build (V4.1 target dir, as everything v41):
```
CARGO_TARGET_DIR=target-v41 nix develop -c cargo build --release -p deepstrix-expertd --features v41
# box 2 (no internet): rsync the tree (exclude target*), then
~/b2-dev.sh cargo build --release --offline -p deepstrix-expertd --features v41     # target-b2/
```

## 2. Wire protocol (`remote_experts::proto`, version 1)

Persistent TCP, one connection at a time per daemon, length-prefixed little-endian frames:

```
header (16 B):  magic u32 = 0x50585344 "DSXP" | version u16 = 1 | kind u16 | seq u32 | payload_len u32
kind 1 HELLO     daemon → client on connect: n_layer, n_expert, n_used, n_embd, xq_bytes_per_token,
                 max_batch, decode_max_b, n_resident, bytes_per_expert, then n_layer × ceil(n_expert/32)
                 u32 ownership bitsets (which experts of which layer the daemon holds)
kind 2 REQUEST   layer u32, b u32, flags u32, n_used u32 (=6), xq_bpt u32 (=5840), reserved u32,
                 t1 u64                  client send stamp (CLOCK_MONOTONIC_RAW), offset 40
                 xq  [b × 5840 B]        Q8_K activation rows (the MoE gate/up input, see below)
                 sel [b × 6] i32         expert ids, -1 = "no pick in this slot" (NO_PICK)
                 ew  [b × 6] f32         router weights (0 for NO_PICK slots)
kind 3 RESPONSE  layer, b, flags, status, t_compute_us, t_server_us, n_embd u32 (=5120), elem_bytes u32,
                 t1 u64 (echoed), t2 u64 (daemon receive), t3 u64 (daemon send)   offsets 48/56/64
                 data [b × 5120] × elem_bytes — weighted partial sums, f16 by default
kind 4 ERROR     status u32 + UTF-8 message (the daemon then closes the connection)
flags bit 0 (REQ_FLAG_RESP_F32): return f32 partials instead of f16.
flags bit 1 (REQ_FLAG_BATCHED):  take the by-expert (prefill) chain even at small B.
```

**HELLO** also carries the daemon's back-to-back `(CLOCK_MONOTONIC_RAW, CLOCK_REALTIME)` pair,
which is what lets a daemon monotonic stamp — and hence its perfetto trace — be mapped onto the
hub's timeline (§5.5).

### 2.1 Clock sync: NTP's estimator over our own link

`t1` client send, `t2` daemon receive, `t3` daemon send, `t4` client receive — t1/t4 on box 1's
CLOCK_MONOTONIC_RAW, t2/t3 on box 2's. Then

```
offset = ((t2-t1) + (t3-t4)) / 2      # add to a box-1 stamp to get box 2's
delay  = ((t4-t1) - (t3-t2)) / 2      # one-way link time
```

i.e. NTP's algorithm over our own link. One message per layer already flows, so a 40-layer token
yields ~40 offset samples at **zero extra traffic**. Why this rather than NTP/chrony: the event to
resolve is a 32 µs RTT against ~0.5 ms layer events, chrony over this link is 100 µs – 1 ms, and
real PTP needs hardware timestamping `thunderbolt-net` does not have. Measured precision: §5.5.

Stamping points matter and are chosen for accuracy, not convenience: `t1` and `t3` are patched
into the already-encoded frame **by the writer thread immediately before `write()`**
(`proto::patch_u64`), so neither frame encoding nor the compute→writer handoff is inside the
sample; `t2` and `t4` are taken by the reader threads the instant the frame is whole. Raw samples
are retained (`ClockSync::samples`, `--clock-dump` writes them as CSV) — the per-sample *delay*
series is how path asymmetry and link queueing become visible instead of being assumed absent.
`CLOCK_MONOTONIC_RAW` rather than `CLOCK_MONOTONIC` so an NTP slew cannot move the clock under a
measurement.

**Activations are Q8_K, not f16/fp8.** The MoE gate/up kernels consume Q8_K (256-element super
blocks: f32 scale + 256 int8 + 16 int16 block sums = 292 B; 20 blocks = **5840 B/token**). The
hub already quantises `ffn_input_norm` to exactly these bytes for its own experts
(`q8k.launch` on the iGPU, and identically on the dGPU for the hot set), so sending them makes
the remote partial **bit-identical to a local computation by construction** — there is no second
rounding on the wire. It is also smaller than f16 (10240 B) and only 10% larger than fp8 +
scales. f16/fp8 activations would have forced a re-quantisation on the remote and a rounding
difference against the local branches.

**Response is f16** (`__float2half`, RNE, done on the device by `f32_to_f16_cast`), 10240 B/token:
at B=1024 that is 10.5 MB vs 21 MB for f32, i.e. 9.5 ms vs 19 ms on the 1.1 GB/s link. The f16
rounding applies to the remote partial only (≤ 2⁻¹¹ relative on that branch, below the Q8/MXFP4
weight noise); `REQ_FLAG_RESP_F32` is there for the bit-identity tests and is cheap at decode
sizes (20 KB). **Rows without a remote pick come back as zeros** (dense response; at a 50% expert
share only 1.6% of prefill rows have no remote pick, so omission would save nothing); the CLIENT
skips the request entirely when no token of the batch has a remote pick (`submit` → `None`).

Sizes: decode B=1: 5.9 KB out / 10.3 KB back (f32: 20.5 KB). Prefill B=1024: 6.03 MB out /
10.49 MB back. Header/pick overhead is 48 B/token.

Framing rules: `seq` is echoed; responses are FIFO (the daemon is single-queue), so the client
may keep several layers in flight; the client rejects a response whose seq/layer/b differ from
its oldest ticket. Buffers are recycled through channels on both sides (no per-request
allocation once warm). Transport recipe from SECOND_BOX.md: TCP_NODELAY, SO_SNDBUF/RCVBUF 4 MB,
TCP_QUICKACK re-armed after every receive, SO_BUSY_POLL 500 µs (plus the flake's
`net.core.busy_read=500`), all applied by `apply_socket_options` (four raw `setsockopt`s,
declared `extern "C"` — no `libc` dependency added).

## 3. Daemon: shard + executor

**`Assignment`** grammar (comma-separated): `L<a>[-L<b>][:<lo>-<hi>]`, `all[:<lo>-<hi>]`,
`<layer>[:<lo>-<hi>]` — e.g. `L20-L32` (whole layers), `L0-L19:192-383` (half of every encoder
layer), `3:0-7,7:0-7` (the loopback test's 16 experts).

**`ExpertShard::load`** allocates one packed `RoutedExpertWeights` pool of `n_experts` slots
(gate/up/down, 18.8 MB per slot in the engine's MXFP4 form — `weight_contract::bytes_per_expert`
checked against the tensor byte size exactly like the pager) and fills it layer by layer, ids
ascending, each layer a contiguous slot range. Reads go through the same
`WeightSrc::read_expert_into` → `V41HfWeights::read_expert_raw` (pread + MXFP4 repack) that
`ExpertPager` and `load_experts_packed` use, `--load-threads` parallel readers over
`--load-batch` experts staged per device copy (3 roles × batch × 6.3 MB of host staging). Per
layer it keeps a pointer-stable device remap of **385** entries: owned id → `-(local_slot)-1`,
everything else (including entry 384) → `0`.

Why not `ExpertPager` itself: its dense windows can only pin layers `0..k`, `ensure` pages one
expert at a time on one thread with `POSIX_FADV`-free but serial reads, and it refills a single
remap per call. The daemon wants an arbitrary `(layer, range)` set loaded once with parallel
readers and one remap per layer. Byte layout, geometry checks and kernel contract are the pager's.

**`MoeExecutor::run(shard, layer, b, xq, sel, ew)`** — picks the remote does not own arrive as
`NO_PICK` (-1) and are replaced by **`SENTINEL_EXPERT` = 384**, whose remap entry (`0`, ≥ 0)
makes every mode-0 het-split kernel treat the slot as "the dGPU takes it" (with `cap =
N_EXPERT_USED` so the resident-rank test never overflows). Consequences: every launch has a
fixed shape (graph-capturable later), gate/up zero-fills the slot's `mid`, down/reduce skip it,
**no partials zero-fill is needed** — the M53 no-fill invariant holds. (Padding with -1 instead
would have made `q2_k_reduce_partials_hetsplit` sum a stale partial row: `e >= 0 && remap[e] >= 0`
is false for -1, so `take = true`.) Unowned real ids are rejected host-side with an ERROR frame
rather than silently skipped.

Two launch paths, chosen by `b <= --decode-max-b` (default 4):
* **decode**: per token `moe_gate_up_batch_hetsplit` (grid 2304/8 × 6) → `q8k` (mid) →
  `moe_down_batched_hetsplit`; 3 launches/token, no host sync (the wire already carries Q8_K, so
  the hub's first `q8k` launch is not repeated).
* **batched** (prefill): the production stage-11 chain — `group_count` zero →
  `moe_group_builder_hetsplit` (group ids = local slots, `n_expert = 384`) → `moe_work_items_builder`
  (CHUNK 32) → host readback of `n_work_items` → `mxfp4_pair … kwide` gate/up → `q8k` (mid) →
  `mxfp4 … by_expert_kwide2` down → `q2_k_reduce_partials_hetsplit`.
Then `f32_to_f16_cast` (unless f32 requested) and a D2H copy straight into the response frame.

Scratch at `--max-batch 1024`: 237 MB (xq 6, mid 57, midq 16, partials 126, out 21, out16 10.5,
members 1.5). Threads per connection: socket reader → bounded channel → compute (the thread that
owns the GPU) → bounded channel → socket writer, so a request for layer L+1 is read while layer L
computes and layer L's 10 MB reply is written while L+1 computes.

Per-request timings recorded (`RequestRecord`): frame read, queue wait, H2D, GPU (launch→sync),
D2H(+cast), write; percentiles per batch size are printed at disconnect (`summarize`) and every
`--log-every N` requests; `--verbose` prints one line per request. The response carries
`t_compute_us` (H2D+GPU) and `t_server_us` (frame complete → response handed to the writer) so
the client can split its round trip into link time and daemon time (`RemotePartial::link_us`).

While idle the compute thread issues one trivial launch every `--keep-warm-us` (default 250 µs)
so the iGPU does not downclock between two layers' requests — worth 4x at B=1, see §5.3.

CLI: `deepstrix-expertd --model /weights/dsv4.1f --experts L20-L32 [--listen 0.0.0.0:7431]
[--max-batch 1024] [--decode-max-b 4] [--load-threads 8] [--load-batch 32] [--verbose]
[--log-every N] [--busy-poll 5000] [--keep-warm-us 250] [--sndbuf N] [--rcvbuf N] [--quickack]`.
Box 2 recipe used for every number below:
```
~/b2-dev.sh ./target-b2/release/deepstrix-expertd --model /weights/dsv4.1f \
    --experts L20-L32 --listen 0.0.0.0:7431 --load-threads 8 --busy-poll 5000 --keep-warm-us 250
```

## 4. Hub client (`RemoteExpertClient`)

```rust
let mut rc = RemoteExpertClient::connect("10.99.0.2:7431", &SocketOptions::default())?;  // reads HELLO
rc.owns(layer, expert_id)              // ownership from HELLO; rc.info() has the bitsets
// at router time (per layer), xq = the hub's d_xq_q8k bytes for the b tokens, sel/ew = router picks:
let ticket = rc.submit(layer, b, &xq, &sel, &ew, /*resp_f32=*/false)?;   // Option<Ticket>; None = no remote pick
// ... issue local iGPU picks + dGPU shared/hot ...
if let Some(t) = ticket { let p = rc.wait(t)?; /* p.f16(): &[u16] b×5120 */ ...; rc.recycle(p); }
```
`submit` masks non-owned picks, encodes the frame into a recycled buffer and hands it to a
writer thread (returns in ~µs at decode sizes; the 6 MB memcpy at B=1024 costs ~0.5 ms);
`wait` blocks on the reader thread's channel. Several tickets may be outstanding (FIFO).

## 5. Measurements

### 5.1 Loopback, box 1 (production V4-Flash server sharing the same iGPU)

`remote_experts_loopback` (16 experts: L3/L7 × ids 0..8, 0.30 GB, load 0.3 s): **every case
bit-identical** — decode B=1 and B=4 with 1/3/6 remote picks on two layers, batched B=64 and
B=1024, f32 and f16 responses, against a reference that loads the same experts into its own
dense slot==id buffer with its own remap and its own launch sequence. Plus:
* remote(3 picks) + local(3 picks) vs the plain 6-pick MoE: max rel diff **1.03e-7** (fp32
  reassociation only) — the split is exact, so the hub can add the remote partial at the combine.
* the f16 response matches a CPU RNE cast of the f32 result on every element (0 mismatches).
* client-side masking: unowned ids dropped, an all-unowned batch sends nothing.
Whole test 5.5 s.

| B | rtt p50 | daemon compute p50 | ms/layer |
|---|---|---|---|
| 1 | 367 µs | 323 µs | 0.38 |
| 4 | 1904 µs | 1373 µs | 1.96 |
| 1024 | 38.0 ms | 34.4 ms | 38.5 (depth 2: 36.2) |

### 5.2 Cross-box: daemon on box 2, client on box 1

Daemon share **L20–L32 = 4992 experts = 93.85 GB**, `--max-batch 1024`, `--decode-max-b 4`.

**Load time and memory.** Checkpoint open 1.0 s; **93.85 GB loaded in 29.6–36.3 s at 2.6–3.2 GB/s**
(8 reader threads, 32 experts staged per device copy, warm page cache; ~2.2–2.4 s per 384-expert
layer). Host **RSS 0.29 GB** (the pool is device memory; host staging is 3 × 32 × 6.3 MB),
**GTT used 94.27 GB** of box 2's 128 GiB — i.e. the pool is 99.6% of the bytes, nothing else is
resident, and the whole 289 GB expert corpus would need ~3 such boxes (as PLAN §4 assumes).
Executor scratch at max_batch 1024: 237 MB.

**Bit-identity across the boxes**: `--check-layer {20,24,27} --check-n 4` loads 4 of the daemon's
experts onto box 1's iGPU and compares — **f32 and f16 bit-identical at B=1, 4 and 64** every
time. Same kernels, same gfx1151, same Q8_K bytes on the wire ⇒ no tolerance needed.

Per-layer round trip (client busy-poll 5000 µs, `--gap-us 1000` = a realistic 1 ms of hub
attention between layers; `link = rtt − daemon server time`):

| B | out | in | rtt p50 | rtt p90 | daemon srv p50 | of which GPU | link p50 | ms/layer | link MB/s | % of 1.1 GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (3 picks) | 5.9 KB | 10.3 KB | **436 µs** | 464 | 390 | 370 | **45 µs** | 0.44 | 360 | (latency-bound) |
| 2 | 11.8 KB | 20.5 KB | 876 µs | 909 | 767 | 740 | 110 µs | 0.89 | 294 | — |
| 3 | 17.7 KB | 30.8 KB | 1207 µs | 1233 | 1091 | 1064 | 110 µs | 1.23 | 441 | — |
| 4 | 23.6 KB | 41.0 KB | 1571 µs | 1612 | 1445 | 1418 | 125 µs | 1.60 | 517 | — |
| 4 (batched path) | 23.6 KB | 41.0 KB | 1368 µs | 1395 | 1239 | 1212 | 129 µs | 1.39 | 501 | — |
| 256 | 1.51 MB | 2.62 MB | 61.1 ms | 62.8 | 32.7 | 32.5 | 28.5 ms | 59.6 | 145 | 13% |
| 1024 | 6.03 MB | 10.49 MB | **74.3 ms** | 75.9 | 51.7 | 51.1 | **22.8 ms** | 80.0 | **724** | **66%** |
| 1024, depth 2 | " | " | 104.9 ms | — | 80.3 | 51.3 | 23.2 ms | 64.0 | 712 | 65% |
| 1024, 48-expert pool | " | " | 57.0 ms | 58.0 | 34.6 | 34.0 | 22.5 ms | 61.3 | 733 | 67% |
| 1024, 6 picks | " | " | 114.5 ms | 135.6 | 75.0 | 74.4 | 39.6 ms | 115.8 | 417 | 38% |

Reading these:
* **Decode (B=1) costs 436 µs per layer, of which the link is only 45 µs.** 40 layers = 17.4 ms
  serialised, but 39 of those 40 RTTs sit under the hub's own attention + local experts (§7b
  rule 2), so what the schedule must actually hide is the *remote branch length*, 436 µs against
  a ~0.55 ms dGPU attention chain — it fits. The 45 µs link time matches SECOND_BOX.md's 32–67 µs
  ping-pong at 4–20 KB, i.e. the transport recipe is already at the measured floor.
* **The daemon's GPU time dominates decode**, not the link: 370 µs for 3 experts = ~123 µs/expert
  = 56.4 MB / 123 µs ≈ **153 GB/s**, ~72% of the iGPU's measured 214 GB/s expert bandwidth, with
  three serialised launches per token. `--picks 1/2/6` gives 177/279/650 µs of GPU — linear in
  picks with a ~60 µs floor, confirming it is weight bandwidth, not fixed cost.
* **Prefill (B=1024) is 74 ms per layer, 66% of the link's 1.1 GB/s ceiling** on 16.5 MB of
  traffic (ideal 15.0 ms vs measured 22.8). Half-duplex in time today: the request finishes
  before the reply starts. Full duplex (separate rings, §SECOND_BOX) would overlap them; that
  plus pipelining layer L+1's request under layer L's compute is the lever, not the encoding.
* **B=2..4 scale linearly (decode path runs tokens one at a time)**; forcing the by-expert chain
  at B=4 (`REQ_FLAG_BATCHED`) is 13% faster (1212 vs 1418 µs GPU), so a DSpark verify batch
  should set that flag — the crossover is at B≈3.
* GB/s in the tool's last column is whole-benchmark throughput including compute, not link rate;
  the "link MB/s" above is the honest transport number.

### 5.3 One real trap found and fixed: the idle iGPU downclocks

With the daemon blocking on `recv` between requests, inserting a realistic 1.5 ms gap between a
reply and the next request **nearly tripled the round trip** — B=1 rtt 436 → 2125 µs, daemon GPU
372 → 1043 µs — because the iGPU drops clocks the moment it goes idle, and the next launch pays
the ramp. This would have hit production exactly (the hub spends ~1 ms per layer on attention
before the next remote request) and is invisible to a back-to-back benchmark.

Fix: `ServeOptions::keep_warm_us` (default 250 µs) — while idle the compute thread issues one
trivial Q8_K launch + sync per period. After it:

| gap between reply and next request | 0 | 700 µs | 1500 µs | 3000 µs |
|---|---|---|---|---|
| rtt p50 before | 436 µs | 560 | 2125 | 2296 |
| rtt p50 **with keep-warm** | 436 µs | **490** | **495** | **472** |

`--keep-warm-us 0` turns it off.

### 5.4 TCP_QUICKACK: measured harmful, now off by default

SECOND_BOX.md's ping-pong recipe includes TCP_QUICKACK. Re-armed after every receive (Linux
clears it after a few exchanges, so "set once" is effectively off) it cost **+260–300 µs per
round trip on replies ≥ 16 KB** — B=1 f32 (20.5 KB back) 750 µs with, 472 µs without; B=4 1865 →
1759 µs — while making no difference at 10 KB. Default is now `quickack: false`; `--quickack`
re-enables it. Explicit 4 MB SO_SNDBUF/SO_RCVBUF and SO_BUSY_POLL both matter as documented
(autotuned buffers cost +290 µs at 20.5 KB).


### 5.5 Clock sync between the boxes — MEASURED

Every request/response carries the NTP quadruple, so each bench run is also a clock measurement.
`--clock-dump FILE` writes every raw sample as CSV.

**First: the two boxes' wall clocks are 35.8 HOURS apart** (box 1 `Sun Sep 13 09:59 UTC`, box 2
`Mon Sep 14 21:46 UTC`). So merging the two perfetto traces on CLOCK_REALTIME is not "accurate to
100 µs – 1 ms", it is off by a day and a half. The measured offset is not a refinement here, it is
the only thing that makes a merged timeline possible at all. (It also silently broke the box-2
build: rsync preserves mtimes, box 2's clock is ahead, so cargo judged every synced file older
than its own artifacts and kept stale rlibs. `b2_run_expertd.sh` now `touch`es the tree first.)

**Loopback sanity (one box, so the true offset is exactly 0):** over 482 exchanges the estimator
reports **offset p50 −1.27 µs** (an earlier run: +0.78 µs). That is an end-to-end check of the
estimator *and* the stamping points, not just of the plumbing; it is asserted in
`remote_experts_loopback` along with `t1 ≤ t2 ≤ t3 ≤ t4` on every sample.

**Cross-box, decode-sized exchanges (B=1), quiet boxes, 2006 samples over 3.0 s:**

| quantity | value |
|---|---|
| one-way delay `((t4-t1)-(t3-t2))/2` | **p50 22.3 µs**, p90 25.6, p99 46.0, min 16.1 |
| relative clock RATE (box 2 vs box 1) | **−64 to −68 ppm**, i.e. the offset genuinely moves ~65 µs every second |
| offset scatter about the fitted drift line | stdev 11.4 µs, p1..p99 spread 20.8 µs, p10..p90 ≈ ±3 µs |
| **operational error** (sample vs the trailing-window median the engine would use) | see sweep below |

The 22.3 µs one-way delay is exactly half SECOND_BOX.md's 4 KB busy-poll ping-pong RTT (32–45 µs),
independently confirming both numbers.

**The window sweep is the load-bearing result.** A trailing median lags by ~half a window, and at
65 ppm that lag *is* the error floor — nothing to do with measurement noise:

| window (samples) | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| \|err\| p50 (µs) | **1.11** | **1.62** | 3.08 | 6.17 | 12.25 | 24.41 | 49.00 |
| \|err\| p90 (µs) | 3.84 | 4.51 | 6.13 | 8.97 | 15.39 | 27.57 | 51.96 |

The default window was 256 (12 µs median error) and is now **32**: |err| p50 **1.6 µs**, p90 4.5 µs
— the "few microseconds" the design needs, against 0.5 ms layer events and a 32 µs RTT. So
"did box 2's partial arrive before the combine wanted it" is an answerable question.

Under load (box 1 running the server plus two other agents' work) the same measurement degrades to
|err| p50 19 µs / p90 74 µs / p99 1965 µs — the tail is scheduler preemption of the stamping
threads, not the link. Worth re-measuring on a quiet box before trusting a tail.

**Path asymmetry is real and visible, exactly as the raw series is meant to show.** NTP's estimator
assumes symmetric paths and its error is half the asymmetry. A B=1024 exchange sends 6.03 MB and
receives 10.49 MB — 4.5 MB of asymmetry — and its offset samples come out **biased by −8.25 ms**
with delay p50 12.7 ms, against ±3 µs at B=1:

| B | offset p50 (relative to B=1) | offset p1..p99 spread | delay p50 |
|---|---|---|---|
| 1 | 0 (reference) | 189 µs | 22.3 µs |
| 4 | −274 µs | 338 µs | 46.0 µs |
| 1024 | **−8255 µs** | 114 ms | 12.7 ms |

Consequence, now enforced in code: `ClockSync::offset_max_b` (default 4) keeps prefill-sized
exchanges in `samples` — their delay series is the asymmetry evidence — but **excludes them from
the published offset**. Without that filter the mixed-traffic offset spread was 151 ms.

API: `client.clock()` → `ClockSync::{offset_ns, delay_ns, drift_ppm, residual_ns, spread,
samples, perfetto_shift_ns, summary}`; `RemotePartial::clock` carries that exchange's own sample.

### 5.6 Perfetto tracing on box 2

`--trace FILE --machine NAME` makes the daemon emit the same raw-protobuf format the engine uses
(`het/perfetto.rs`), via a new additive `TrackExporter` there (the existing `DeviceTimingExporter`
hardcodes the hub's four dGPU+iGPU streams; box 2 has one iGPU and also wants host tracks).
Verified output — 26004 packets, 1.0 MB, parsed back with an independent decoder:

| track uuid | name | slices |
|---|---|---|
| `0x424f5832_00000001` | `lumi-brain2 igpu.compute (device)` | 4004 — device time from a HIP event pair per request, e.g. `moe L20 B=1` |
| `0x424f5832_00000002` | `lumi-brain2 igpu.xfer (device)` | 0 (declared; the daemon's H2D/D2H are synchronous today) |
| `0x424f5832_00000010` | `lumi-brain2 expertd.request (host)` | 4004 — e.g. `L20 B=1 decode` |
| `0x424f5832_00000020` | `lumi-brain2 expertd.ssd (host)` | 4992 — **one span per expert read, labelled `ssd L{layer} E{expert}`**, so a stall attributes to a specific pick |

uuids are disjoint from the hub's (`0x44504755_*` / `0x49504755_*`) so the two files merge into one
timeline in trace processor. The device track re-anchors every 512 requests
(`ServeOptions::re_anchor_every`) to bound GPU/host drift, matching the engine's per-token
`re_anchor`. Timestamps are box 2's CLOCK_REALTIME; **apply `ClockSync::perfetto_shift_ns` (the
bench prints it) to place them on box 1's timeline** — with the 35.8 h wall-clock gap this is
mandatory, and it is as accurate as the offset (µs) rather than NTP's ms.

The SSD spans currently cover the *load* reads (the daemon serves exactly its assignment, so there
is no serve-time miss path yet); when the cold tier lands they are the same spans on the same track.

## 6. What remains to integrate (proposed call sites)

Nothing in `forward_layer.rs` / `forward_prefill.rs` was modified. The hub needs, per layer,
(a) to send its picks + Q8_K activations to the remote at router time, (b) to exclude the
remote's picks from its own iGPU (and dGPU-hot) legs, (c) to add the remote partial at the
combine. Concretely:

1. **Engine state**: `HetModelState` (or `WorkerState`) holds `Option<RemoteExpertClient>` and a
   per-layer `remote_owned: Vec<[u32; 12]>` (from `client.info().owned`).
2. **Exclusion (b) without a new kernel**: the iGPU leg already skips every pick whose remap
   entry is `>= 0` (mode 0). So the hub's iGPU remap for layer L gets `remap[e] = 0` for every
   remote-owned `e` (pager LRU path: in `ExpertPager::ensure`'s reset loop; dense-window prefill
   path: pass a remap with `-(e)-1` for local ids and `0` for remote ids and take the
   `launch_hetsplit` + `launch_reduce_partials_hetsplit` branch that `moe_remap.is_some()`
   already selects) **with the cap pinned at `N_EXPERT_USED`** so the rank test cannot spill a
   remote pick back to the iGPU. If a dGPU hot set is also in play, the dGPU (mode 1) must use a
   remap where remote ids are NEGATIVE (not its slots) — i.e. two remaps, one per device, both
   derived from the same ownership table. With V4.1's current `hot_experts = None` it is one remap.
3. **Send (a)**: in `forward_layer_standalone_graphs_paged` right after the router's `d_selected`
   readback (the pager already reads it back — `sel_host`), and before `pg.ensure`:
   ```rust
   // d_xq_q8k is produced on the iGPU by ie.q8k.launch(...) inside the routed_moe graph today;
   // for the remote it must be produced BEFORE the graph: one q8k launch on ffn_input_norm_recv
   // (or the dGPU's moe_xq, identical bytes) + a 5840 B D2H copy.
   let ticket = remote.submit(layer as u32, 1, &xq_host, &sel_host, &ew_host, false)?;
   ```
   Prefill (`forward_layer_batch_v2` stage 11): after `bd.d_selected/d_ew` are on the host side of
   the router (the batched path today never reads them back — one `copy_to_host` of `b×6` i32 +
   f32 per layer, 48 KB at B=1024) and `si.d_xq_q8k` is quantised: `remote.submit(layer, b,
   &xq_host, &sel_host, &ew_host, false)` — before the local iGPU chain is issued so the 6 MB
   leaves while the iGPU works.
4. **Combine (c)**: `wait(ticket)` just before `ffn_combine`; the f16 rows go into a new dGPU
   buffer `ffn_moe_remote16: DeviceBuffer<u16>` (`[B × 5120]`, ~10 MB at B_MAX) and the combine
   adds `f16→f32(ffn_moe_remote16) + ffn_moe_recv (+ ffn_moe_dgpu)`. That is one extra operand in
   the existing vec-add / hc_post prologue; the cheapest exact way is a tiny `f16_to_f32_add`
   kernel or to pass the f16 pointer to the combine kernel (both < 30 lines; not written here).
   The wait sits after the local legs are issued, so the remote's RTT overlaps them (§7b rule 2).
5. **Signature exposed** (the whole hub-side surface):
   ```rust
   impl RemoteExpertClient {
       fn connect(addr: impl ToSocketAddrs, opts: &SocketOptions) -> Result<Self>;   // reads HELLO
       fn info(&self) -> &proto::ShardInfo;                 // ownership bitsets, max_batch, decode_max_b
       fn owns(&self, layer: u32, e: i32) -> bool;
       fn submit(&mut self, layer: u32, b: usize, xq: &[u8], sel: &[i32], ew: &[f32],
                 resp_f32: bool) -> Result<Option<Ticket>>;                 // None = no remote pick
       fn submit_flags(&mut self, layer: u32, b: usize, xq: &[u8], sel: &[i32], ew: &[f32],
                       flags: u32) -> Result<Option<Ticket>>;               // REQ_FLAG_{RESP_F32,BATCHED}
       fn wait(&mut self, t: Ticket) -> Result<RemotePartial>;              // FIFO
       fn recycle(&self, p: RemotePartial);                                  // buffer reuse
       fn call(&mut self, ..) -> Result<Option<RemotePartial>>;              // submit + wait
       fn in_flight(&self) -> usize;
   }
   impl RemotePartial {
       fn f16(&self) -> &[u16];      // b × 5120, row-major, zeros where no remote pick
       fn f32(&self) -> &[f32];      // when REQ_FLAG_RESP_F32 was set
       fn bytes(&self) -> &[u8];     // raw payload, for a direct H2D into the combine buffer
       fn rtt_us(&self) -> u32; fn link_us(&self) -> u32;
       // plus t_remote_compute_us, t_remote_server_us, layer, b
   }
   ```
   `xq` is `b × 5840` bytes = exactly the hub's `d_xq_q8k` for those tokens; `sel`/`ew` are the
   router's `b × 6` arrays. `bytes()` is the zero-copy path: the f16 payload can go straight into
   a `DeviceBuffer<u16>` with one `copy_from_host` and never be touched by the CPU.

6. **Clock + trace hooks the schedule work will want**: `client.clock().offset_ns()` converts a
   box-2 stamp to box-1 time, so a remote partial's `t3` (daemon send) can be compared directly
   against when `ffn_combine` actually wanted it — that is the per-track busy-fraction question
   PLAN §7b.5 makes the acceptance metric. Run the daemon with `--trace` and merge its file with
   the hub's, shifted by `clock().perfetto_shift_ns()`.

Placement per §7b rule 1 (balance the branches by latency, hottest experts local, tail remote)
is a policy on top of `Assignment`: today the daemon takes whole layers / ranges from the
command line; a placement file (expert ids per layer, like `DGPU_HOT_EXPERTS_FILE`) is a
straightforward extension of `Assignment::parse`.

## 7. Known limits / next levers

* Decode path serialises tokens (B=4 = 4 × B=1); a DSpark verify batch at B=3–4 should go
  through the batched path or a graph-captured 4-token decode chain — measure with
  `--decode-max-b 1`.
* Fixed per-request cost at B=1 (~140 µs on the loopback: three sync `hipMemcpy`s + three
  launches + sync + cast + D2H) is the next thing to cut on the daemon: pinned host staging +
  async copies, a captured graph per layer (all shapes are fixed), and returning f32 at B=1 to
  skip the cast.
* One connection at a time; a second hub request stream would need a second daemon or a
  per-connection executor.
* Misses: the daemon serves exactly its assignment; a pick it does not own is an error, not a
  page-in. The SSD tier on box 2 (PLAN §3.3) would hang off `ExpertShard` as an LRU over a spare
  slot range — not built. Box 2's plaintext NVMe measures 8.60 GB/s at 8 threads vs box 1's
  encrypted 4.31 GB/s, so the cold tail belongs here, behind this daemon, not on the hub.
* Prefill at B=1024 is half-duplex in time (request drains, then reply). Overlapping the two
  directions and issuing layer L+1's request under layer L's compute is worth ~7 ms/layer.
* The daemon reads a 6 MB request in 8.8 ms (685 MB/s) but a 1.5 MB one in 20.8 ms — small
  prefill batches are not reaching link rate; unexplained, worth a look before B≈256 chunks matter.

## 8. Placement note after CED (2026-09-14)

CED prefill (now default) touches only the **20 encoder layers** plus a 128-token decoder replay,
so the prefill-relevant expert set is ~144 GB, not 289 GB. The share measured above (L20–L32,
93.85 GB) is a *decoder* share: it exercises the machinery but contributes nothing to prefill.
For the two-box prefill target the daemon's assignment should be encoder-weighted — e.g.
`--experts L0-L19:192-383` (half of every encoder layer = 3840 experts = 72 GB) leaving the other
half to the hub's ~55 GB pool, which covers the whole encoder set across the two boxes with room
to spare. That is a command-line change only; `Assignment::parse` already accepts it. Decode still
needs all 40 layers, so the steady-state assignment is a per-layer expert list (the placement file
extension noted in §6), not a layer range.
