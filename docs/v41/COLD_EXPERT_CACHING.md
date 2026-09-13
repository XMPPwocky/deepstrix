# Cold-expert residency: static placement vs dynamic caching — MEASURED 2026-09-11

**Bottom line: dynamic (LRU) residency beats static frequency placement by 11-48× on SSD reads
per token, which is what makes a fully-native (zero quality loss) V4.1 configuration viable on
two boxes.** Measured on V4-Flash as a locality proxy; V4.1 numbers are extrapolated.

## Why this was measured

V4.1 native needs ~276 GiB of expert weights; two boxes give ~209 GiB. The gap is either
requantized (quality cost, see PLAN §3) or streamed from SSD (speed cost). The speed cost was
estimated at ~9 SSD reads/token from the *aggregate* production expert histogram. That estimate
was wrong in both directions, and the fix is a policy change, not more hardware.

## Method

New hook `DEEPSTRIX_EXPERT_TRACE=<path>` (engine.rs, decode path, beside the existing
`DEEPSTRIX_EXPERT_STATS`): dumps per-token, per-layer chosen expert ids as u16 LE,
N_EXPERT_USED per layer × N_LAYER per token, in decode order. The histogram that was already
there throws the *sequence* away, which is exactly what a caching question needs.

Trace: 4,672 decode tokens on the production server (Vision-Exp UD-IQ3_XXS, 43 layers,
256 experts, top-6 = 258 picks/token), two conversations:
- **A** (2,560 tok): English technical prose + Rust code (CoW filesystem internals).
- **B** (2,112 tok): Chinese history exposition + classical poetry with prosody analysis.

"Cold" is ranked by the **production** decode histogram (1.79M tokens of real traffic), not by
the trace itself. Caches are warmed before measurement.

### Two confounds hit on the first pass (both fixed; recorded so they are not repeated)
1. **Circular cold definition.** Ranking slots by the trace's own frequencies makes "coldest 25%"
   mean "slots this trace barely used", so their pick share is ~0 by construction. It reported
   0.66 cold picks/token; the true figure with an independent ranking is 21.7.
2. **Cold-start misses counted as LRU misses.** Over a short trace, first-touch dominates and
   made LRU look 5× *worse* than static. With a 1,000-token warm-up, LRU is 11-48× better.

## Results

### 1. A static hot set derived from aggregate stats does not fit any individual conversation

Misses per token (= SSD reads) with the top-M slots by production frequency resident:

| resident | conversation A (EN technical) | conversation B (ZH history/poetry) |
|---|---|---|
| 75% | 21.3 | 63.3 |
| 85% | 8.9 | 40.0 |
| 90% | 4.3 | (not run) |

The aggregate histogram is dominated by the traffic mix the server usually sees. A conversation
in another language misses 4.5× more. **"Globally cold but locally hot" is real and large.**
Working sets: A touches 9,323 slots, B touches 9,595, sharing 8,430. **1,165 slots are unique to
the Chinese conversation** — the concrete form of the language/domain expert cluster.

### 2. LRU at the same memory budget

Misses per token, same residency budget, cache warmed:

| resident | segment | static | LRU | gain |
|---|---|---|---|---|
| 75% | A steady state | 21.34 | 1.45 | 14.7× |
| 75% | first 300 tok after topic switch | 60.22 | 5.39 | 11.2× |
| 75% | B steady state | 63.31 | 1.71 | 37.0× |
| 85% | A steady state | 8.93 | 0.83 | 10.8× |
| 85% | first 300 tok after topic switch | 39.20 | 3.08 | 12.7× |
| 85% | B steady state | 39.95 | 0.84 | 47.6× |

A topic switch costs a short burst (3.1 misses/token for ~300 tokens) and then settles back.

### 3. Misses are predictable from very recent history

Of the static-placement misses at 85% residency: 50% were also picked at token t-1, 72% within
the last 8 tokens, 84% within the last 32. So what is not cached is largely prefetchable, and
the two mechanisms compose.

### 4. Static + LRU victim cache (for reference)

85% static + a victim cache of C slots: C=512 (4.3 GiB) gives 0.29 (A) / 1.22 (B) reads/token;
C=1024 (8.7 GiB) gives 0.16 / 0.25. Pure LRU at a *smaller* total budget is competitive, because
static spends slots on globally-hot experts this conversation never touches.

## What it means for V4.1

At 209 GiB of 276 GiB native = **75.7% resident**, scaling the 75% row by picks/token
(240 for V4.1 vs 258 here): **≈1.4 SSD reads/token**, versus ~20 under static placement.
At 18.8 MB and 2.7 ms per native expert read, that is ~3.8 ms serial, less with prefetch.

| configuration | decode | quality loss |
|---|---|---|
| all-native, static placement | ~54 ms of SSD → ~13 tok/s | 0 |
| all-native, LRU residency | ~30 ms → **~33 tok/s** | 0 |
| uniform rotated-LDLQ IQ3_S | ~27 ms → ~37 tok/s | ~2-3% |

**The zero-quality-loss configuration is now within ~10% of the quantized one.** That changes
PLAN §3a's verdict and lowers the value of a third box.

### Implementation note: this may be free

Dynamic residency over mmap'd expert weights is what the **OS page cache already does**. If the
native expert file is mmap'd and the resident set is left to the kernel, we get LRU without
writing a cache. Open questions: 4 KiB page granularity vs 18.8 MB experts (needs readahead or
`MADV_WILLNEED` on the whole expert extent), interaction with the ~85 GiB of genuinely pinned
weights, and whether GTT-mapped iGPU access can be served from page cache at all. If not, an
explicit LRU over expert slots is a few hundred lines and the trace says it is worth it.

## Caveats
- Measured on **V4-Flash** (256 experts/layer, 43 layers). V4.1 has 384 experts and 40 layers;
  a wider expert pool could enlarge the working set. Re-measure once V4.1 runs.
- Two conversations, one switch. **Agentic traffic with interleaved tool calls is untested** and
  is the case that matters most for this user; it may thrash more than long monologues.
- Does not contradict [[hot-expert-placement-inert]]: that result is about dGPU placement when
  every expert is already RAM-resident, so misses cost a slower read, not an SSD trip. Policy
  only becomes load-bearing once a tier genuinely does not fit.

## Reproduce
`DEEPSTRIX_EXPERT_TRACE=<path> ~/run_deepstrix.sh --bg`, generate, then
`analyze3.py` (scratchpad). The hook syncs every layer: ~17-26 tok/s instead of ~29. Diagnostic
only, off by default.

---

## Addendum 2026-09-12 — the 2.7 ms miss cost was ~2x optimistic, and needs a parallel reader

Phase-A prototype (`tests/bench_expert_miss_cost.rs`) measures what one cold miss actually costs:
read an expert-sized extent from an **actual V4.1 shard on the real target volume** (LUKS-encrypted
`/persist`), then DMA to the iGPU. Measured with the V4.1 download running concurrently, so these
are pessimistic on contention and optimistic on nothing.

| 18.8 MB (V4.1 native expert) | read ms | H2D ms | total ms | eff GB/s |
|---|---|---|---|---|
| cold, single synchronous pread | 12.4-18.2 | 2.6-4.7 | **17.0-20.9** | 0.90-1.11 |
| warm (page-cache hit) | 1.1-1.9 | 2.6 | 3.7-4.5 | 4.2-5.1 |

Cold read vs reader parallelism (read only, no H2D):

| threads | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| cold ms | 7.80 | 5.91 | 5.81 | 5.40 | **5.13** |
| eff GB/s | 2.41 | 3.18 | 3.24 | 3.48 | **3.66** |

**Conclusions:**
1. **The plan's assumed 2.7 ms/miss is not achievable; ~5.1 ms is the floor here** (3.66 GB/s, not
   the 7 GB/s of the device datasheet). LUKS decryption sits on the read path and the volume was
   simultaneously being written by the download.
2. **A naive synchronous `pread` costs 12-18 ms and WOULD kill the design** (1.4 misses/token x 18 ms
   = 25 ms/token). **The miss path must be parallel/async (io_uring or a thread pool) — this is now
   a hard engineering requirement, not an optimisation.**
3. **The H2D copy (~2.6 ms) is pure waste on Strix** — the iGPU's "device memory" IS system memory.
   Mapping page-cache pages into GTT directly would remove it. That is phase B and it is worth
   ~2.6 ms of every miss.
4. **A page-cache HIT costs ~1.1-1.9 ms of read + the avoidable copy** — i.e. an LRU *hit* is nearly
   free if mapping works, which is what §3.3's model assumed.

**Revised decode impact** (1.4 misses/token, parallel reader, GTT mapping):
serial 1.4 x 5.1 = 7.1 ms; with ~50% prefetched ≈ 3.6 ms. All-native decode becomes
**~27-30 tok/s** rather than the 32 in PLAN §7 — still far above static placement's ~13, and still
within reach of uniform IQ3_S (~37). **The conclusion survives; the margin is thinner.**

**Caveats:** measured under download contention; LUKS is on the path; 3.66 GB/s at 16 threads may
be CPU-bound on decryption rather than the device ceiling — re-measure once the download finishes.

### Why the cold read is slow — diagnosed 2026-09-12 (it is NOT the cipher, and NOT compression)

Storage layout: one 1.9 TB NVMe → `nvme0n1p2` LUKS → `cryptroot` **btrfs**, `compress=zstd:3`,
everything (incl. `/persist/lumi`) inside it. `/boot` (1 GB vfat) is the only unencrypted mount.

| suspect | verdict | evidence |
|---|---|---|
| AES cipher | **innocent** | `cryptsetup benchmark`: aes-cbc decrypt **7794 MiB/s** single-threaded, AES-NI present |
| btrfs zstd compression | **innocent on READ** | apparent size == on-disk size (6.9 G / 6.9 G) → weights incompressible (cf. the 3.90 bits/weight entropy result), so btrfs stores the extents raw and never decompresses. Costs write CPU during download only |
| fragmentation / COW | **contributing** | 2504 extents for one 6.9 GB shard (~2.8 MB avg vs ~55 extents if laid out at btrfs's 128 MB max). A concurrent multi-worker downloader produces exactly this |
| **dm-crypt workqueue** | **PRIME SUSPECT** | throughput never plateaus: 1→128 threads gives 2.30→**4.60 GB/s** and is still climbing. That is a *latency*-bound path, not a bandwidth ceiling — the dm-crypt signature (every bio queued to a kernel workqueue instead of processed inline) |

**Fixes, in order of value:**
1. **Bypass the dm-crypt workqueues.** Live + reversible:
   `cryptsetup refresh cryptroot --perf-no_read_workqueue --perf-no_write_workqueue`
   (add `--persistent` to keep it). NixOS declarative:
   `boot.initrd.luks.devices.cryptroot.bypassWorkqueues = true`. Typically 2-3x on NVMe read
   latency, moves no data. **If it delivers that, the miss drops ~5 ms → ~2 ms, at or under the
   plan's original 2.7 ms assumption, and all-native returns to ~32 tok/s.** Needs root.
2. **Defragment the weight files**: `btrfs filesystem defragment -r -t 1G <dir>`. Safe, helps
   readahead. Secondary to (1).
3. **Turn compression off for the weights dir** (`btrfs property set <dir> compression none`) —
   no read benefit (already raw) but speeds the remaining download. `chattr +C` additionally
   disables COW+checksums, but only for newly created files.
4. **Plaintext partition — NOT on this box.** Single drive, one LUKS partition, no free space;
   would require shrinking btrfs → LUKS container → partition on a live root. Bad risk/reward.
   **Do it on the SECOND BOX instead** (arrives ~2026-09-14, installed fresh): put the weights on
   a separate plaintext partition. They are public HuggingFace downloads — no confidentiality
   argument — and arranging it at install time is free.

### Workqueue bypass APPLIED and measured 2026-09-12 (`--perf-no_read_workqueue`)

Same file, same parameters, same background load (download writing, server up):

| path (18.8 MB cold) | before | after | change |
|---|---|---|---|
| single synchronous read | 12.35 ms | **7.60 ms** | −38% |
| same + H2D copy | 17.01 ms | **10.70 ms** | −37% |
| 1 thread | 8.17 | **6.88** | −16% |
| 16 threads | 5.41 | **5.21** | −4% |
| 128 threads | 4.09 | **3.68** (5.11 GB/s) | −10% |
| warm read | 1.08 | **0.70** | −35% |

**Reading of the result:** the gain is concentrated where the queue hop is unhidden — a single
synchronous read gains 38%, 128 threads only 10%, because concurrency was already hiding the
latency. **So the bypass does NOT remove the "miss path must be parallel" requirement; it makes
the naive path much less catastrophic.**

**Revised again:** miss floor 4.09 → **3.68 ms**. At 1.4 misses/token with ~50% prefetched that is
~2.6 ms (vs the 1.9 ms the plan originally assumed at 2.7 ms/miss) → decode ~31 ms →
**all-native back to ~32 tok/s**, i.e. PLAN §7's original figure stands.

**NEW top lever: the H2D copy.** At 128 threads a cold miss is 3.68 ms read + 2.6 ms copy — the
copy is now **41% of a miss**, and on Strix it is pure waste because the iGPU's "device memory" IS
system memory. Mapping page-cache pages into GTT deletes it outright. **Phase B is now worth more
than any further read tuning.**

### Phase B measured 2026-09-12 — the copy is NOT waste, but it can be skipped entirely

**Correction to the phase-A addendum.** I claimed the ~2.6 ms H2D copy was "pure waste on Strix
since the iGPU's device memory IS system memory". **That was wrong.** GPU read bandwidth by source
(18.8 MB, device-to-device copy out of each):

| source | ms | GB/s |
|---|---|---|
| device memory (`hipMalloc`) | 0.048 | 391 (cache-resident; treat as "very fast") |
| host pinned, coherent | 2.606 | 7.21 |
| host pinned, NON-coherent (`hipHostMallocNonCoherent`) | 2.609 | 7.20 |

The coherence flag makes **no difference**, and 7.2 GB/s is exactly the phase-A copy cost. So the
iGPU reads `hipHostMalloc` memory ~50x slower than `hipMalloc` memory even though both are the same
LPDDR5X. The copy is not overhead — it is the price of getting bytes somewhere the GPU reads fast.
Leaving experts in host memory would just move the same 2.6 ms from copy-time to read-time, and
would pay it *on every use* rather than once per load.

**XNACK is unavailable.** `rocminfo`: "XNACK enabled: NO"; `amdgpu.parameters.noretry = -1` (auto,
driver declines); `HSA_XNACK=1` does not change it. GPU page-faulting on an mmap'd file — the ideal
design — is a CDNA feature, not available on gfx1151/gfx1201. Explicit staging is the only route.

**What DOES work: `pread()` straight into device memory.** The `hipMalloc` pointer is CPU-addressable
on this APU (probe: wrote 0xA5 to `dev.raw()`, read it back). So the kernel's `copy_to_user` can land
directly in GTT, with no host staging buffer at all:

| miss path (18.8 MB cold) | ms | GB/s |
|---|---|---|
| naive pread into host + copy | 10.70 | 1.76 |
| parallel (128t) pread into host + copy | 3.68 + 2.61 = **6.28** | 3.0 |
| pread direct into device, 1 thread | 7.67 | 2.45 |
| pread direct into device, 16 threads | 5.92 | 3.18 |
| **pread direct into device, 64 threads** | **4.82** | **3.90** |

**Direct-to-device with a parallel reader is the best measured miss path: 4.82 ms, 23% better than
the two-step**, and the GPU then reads the expert at full device bandwidth. Writing into GTT costs
the CPU ~14% vs writing into host memory (2.45 vs 2.73 GB/s single-threaded), but that is far
cheaper than paying a whole second pass over the data.

**Revised decode impact:** 1.4 misses/token x 4.82 ms = 6.7 ms serial, ~3.4 ms with ~50% prefetched
→ decode ~30 ms → **all-native ~32 tok/s**, i.e. PLAN §7's original figure, now measured rather than
assumed.

**Design consequence:** the expert cache lives in **device memory**, and the miss path is a
**multi-threaded `pread` directly into the destination slot** — no host staging buffer, no
`hipMemcpy`. Prototype: `tests/bench_expert_miss_cost.rs` (`bench_gtt_as_page_cache`,
`bench_pread_into_device`, `bench_pread_into_device_threads`).
