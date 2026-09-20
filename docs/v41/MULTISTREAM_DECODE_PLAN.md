# Multi-stream decode: plan, ceiling, scheduler (2026-09-19, rev 1.2, APPROVED by review at 1.1)

Rev 1 after an adversarial architecture review of rev 0 (31 findings; the ones that
changed conclusions are marked "REV" below); rev 1.1 after the re-review (7 new
findings, verdict APPROVE WITH CHANGES; the changes are marked "REV2"). Model: `scripts/multistream_plan_model.py`
(rev 1). Trace simulation: scratchpad `simms2.py` (rev 2: streams are DISTINCT
REQUESTS, each request's prefill rows replayed through the same pools). Every number
is tagged MEASURED (this hardware), SIMULATED (pick trace replayed against the
production pools), or ESTIMATED (a parameter; section 8 says how it gets measured).

## 0. The short version

* **Batching is worth ~2-3x on the hardware we have.** One step of 16 streams touches
  61 experts per layer where 16 serialized tokens touch 96 (SIMULATED), and the 22 ms
  dGPU chain, the 40 per-layer remote calls and the per-layer host glue are paid once
  per step instead of once per token. Today's code batched as-is: 15 -> 34 tok/s
  DECODE aggregate at S=8, 37 at S=16. What caps it is the expert MISS path: at S=16
  a layer waits on a box-2 read 61% of the time and 43 misses per step land on box
  2's one-expert-at-a-time reader (SIMULATED), 194 ms of a 433 ms step (box 1's own
  142 ms of reads sit under box 2's leg and are hidden).
* **REV2: every throughput number below is decode-only unless it says "+pf".** The
  scheduler runs prefill chunks as separate forwards, so under an agent workload
  the wall is decode PLUS prefill. The trace carries 5.1 prefill rows per decode
  token (MEASURED: 97,559 encoder + 1,186 replay rows per 19,139 decode tokens);
  at ~500 chunk rows/s that is 10 ms per token, which takes today's-code S=8 from
  34 to 26 tok/s and the third-box S=16 from 99 to 49. Prefill is 30-55% of the
  wall on that workload, so the CHUNK path (its misses, encoder residency on box 2,
  the replay offload) is first-order and is in M2's scope, not an aside.
* **REV: the largest software lever after batching is a hits-first MoE.** Launch the
  resident experts' groups, read the misses while they run, launch the missed ones
  into their own partial slots, reduce in the fixed slot order. It hides
  `min(compute, read)` per box per layer with no numeric change and no new kernel:
  +14% at S=8, +27% at S=16, +37% at S=32 over lockstep (model). With the Engram
  gather issued at sample time and the multi-segment busy-poll hold fixed: 43 tok/s
  at S=8, 50 at S=16, 64 at S=32 on today's hardware (decode-only; 30 / 33 / 39 +pf).
* **Hardware, in tok/s per dollar, on top of that (decode-only / +pf):** a 7 GB/s
  plaintext NVMe in BOX 2 alone (48 / 61 / 75, i.e. 99% of the two-drive gain at
  S<=16 — REV2); then box 1's (48 / 61 / 82; 32 / 38 / 45 +pf); a second drive per
  box in RAID0 (56 / 72 / 93); a third 128 GB box, zero misses and a three-way split
  (76 / 99 / 128; 43 / 49 / 55 +pf); the IQ2_S requant reaches the same ceiling on two
  boxes if its quality holds. A Gen3 cable and a second USB4 link are cheap and worth
  4-6% above 16 rows.
* **REV: box 1 at 128 GB is a second-order lever that only pays AFTER the drives.**
  With today's disks the swap is a 17-32% LOSS (it moves misses onto box 1's slower
  dm-crypt drive); with fast plaintext NVMe in both boxes and the expert share
  following the slots it is +7-10% at S=16-32 (SIMULATED + model, 2.4). Total RAM
  and therefore total residency do not change; what moves is where the misses land
  and how the two MoE legs balance.
* **Per-stream speed is the QoS trade.** S=8 gives each stream 4.3 tok/s today, 6.1
  with fast NVMe, 9.4 with a third box — while it is decoding. During a pending
  prefill the same stream gets one step per chunk tick (~1.4 s at S=8, 5.3), so the
  per-stream floor in G2 is a decode-only floor and G3 bounds the prefill share.
* **REV: the batched step is the DSpark verify path, but it is NOT bit-identical to
  today's decode** (it moves the local MoE from the per-token `hetsplit` kernels to
  the by-expert chain, which box 2 measured at 1e-7 agreement, not 0). The fidelity
  gate is therefore (a) bit-exact batch invariance WITHIN the new path and (b) KLD
  against today's decode under the existing bar. DSpark rows pay modestly with fast
  NVMe (S=8: 6.8 vs 6.1 tok/s per stream at E=2.9) and strongly at zero misses.
* **Scheduling today is strict FIFO on one live KV state.** The design holds S live
  states, runs decode steps and prefill chunks as SEPARATE forwards (fusing them would
  make a stream's tokens depend on batch composition), gives prefill a share of steps
  rather than a time budget (a chunk is miss-bound at ~1 s on today's box 1), orders
  prefills shortest-first with an aging bound, and admits by VRAM and a per-stream
  floor. REV: the simulation says the LRU tolerates prefill scans (+8-21% decode
  misses at S>=8), so pool protection is an M2 option, not a precondition.
* **Measure before building (server down ~30-45 min, ask):** section 8, M0.

## 1. Inputs

### 1.1 MEASURED (this hardware, 2026-09-14..19)

| quantity | value | source |
|---|---|---|
| warm S=1 token | 58.5 ms: dGPU chain (`sel_sync`) 22.2, box-2 srv 21.6 hidden behind box 1's post-submit work, local MoE ~12-14, shared expert 3.9, ensure hit path 2.7, host glue ~5 | LINK_IDLE_LATENCY.md 2026-09-19 |
| live S=1 token | 66-78 ms; box-1 misses ~1.2/token at 8.85 ms; box-2 misses inside `srv` (live 26.9 vs warm 21.6) | token summaries |
| box 2 batched path | `srv = 105 + 20*rows + 87*D` us/layer, 216 GB/s per distinct expert (`--pool 3`, may be slightly cache-assisted) | DSPARK_VERIFY_BATCH_MODEL.md |
| box 1 decode MoE kernels | per-token `hetsplit` family, 182 GB/s combined, at the streaming ceiling cache-defeated. Both boxes are gfx1151; the plan runs ONE by-expert family on both | ROADMAP item 3 |
| link | 90 us fixed per call (busy-poll fix) + 1.1 GB/s (4 KB frames, receive-ring bound); cable is Gen2 x2; a reply over one 65,520-B segment is held by the busy-poll loop: B=4 f32 measured 358-367 us vs ~186 modelled | LINK_IDLE_LATENCY.md |
| box 2 miss path | ONE expert at a time (3 role threads per expert, experts sequential in `ensure_layer_inner`); E100 4.47 GB/s, flat across QD | remote_experts.rs:2120-2175 |
| box 1 miss path | `ensure_batched` reads a layer's misses in ONE parallel pass (default on) with 3 role threads each; 8.85 ms/miss live; drive ceiling ~4.3 GB/s (dm-crypt) | expert_pager.rs:1413 |
| Engram | `rows_for` per token on the worker thread: 2 layers x 24 spawned threads x 2 preads, 1.73 ms/token live; a batched twin `rows_for_chunk` exists (0.42-0.93 ms/token at QD 32) | engine_worker.rs:58,114 |
| dGPU VRAM | 12.08 of 17.1 GB used with ONE `HetModelState` at `--ctx 307200` (~5.0 GB free); two-lane prefill scratch ~885 MiB at 512 rows | sysfs; batch_scratch.rs |
| KV state | all KV, compressor and index-key state on the dGPU (`HetModelState::alloc` passes the dGPU for both); derived 1,680 B/token + 47 MB (563 MB at 307K), ROADMAP quotes 895 MB — measure | state.rs:728 |
| host RAM | box 1 87/93 GB + 10 GB swap in use; box 2 111/124 | `free -g` |
| snapshot paths | `restore_ms` 7-508, `reset_ms` 3-132; modes seen: restore 241, full 27, extend 0, exact 0; saves are blocking `copy_to_host` on the worker thread | log; snapshot.rs |
| prefill | encoder 400-700 tok/s; prefill hit 0.88-0.92 on box 1; CED replay is a FIXED 128-row decoder pass, 3.5 s mean (5.4 s last request); `V41_REPLAY_OFFLOAD=1` -> 0.2 s | ROADMAP |
| daemon | one TCP connection at a time; `--max-batch` up to 1024; the by-expert path has a host readback (`n_work_items` -> grid size) with a stream sync inside it; `miss_mask` is filled only for b==1 | remote_experts.rs:2653-2660, 2971 |
| serial multi-row oracle | `V41_VERIFY_DECODE_PATH=1` runs `forward_layer_standalone_graphs_paged` layer-major over B rows with `publish_pos_slot` | engine_worker.rs:3673 |

### 1.2 SIMULATED (simms2.py rev 2)

Streams are distinct requests of the trace (9 requests, 37-4,375 decode tokens; S>8
adds offset copies of the longest runs and is flagged correlated). A stream that
finishes picks up the next request; with `prefill=y` that request's prefill rows
(98,745 rows = 1.97M P lines / 20 layers; 5.1 rows per decode token — REV2, rev 1
said "2.6x" by comparing line counts) are replayed through the same LRU pools
first, as the unified pool sees them. Production slot counts 4,454 / 6,160,
hash share 0.397.

    S  K pf | U/lyr   U1    U2 |  m1%   m2% | m1/step m2/step | P1    P2   | ws1   ws2  | b1 scan/chunk
    1  1  y |  6.0   2.4   3.6 | 0.76  1.17 |   0.73    1.68  | 0.018 0.041 | 1523  2244 | 105
    2  1  y | 11.3   4.4   6.9 | 0.70  1.09 |   1.24    2.99  | 0.030 0.071 | 2207  3329 | 103
    4  1  y | 20.7   8.1  12.6 | 0.75  1.25 |   2.44    6.30  | 0.059 0.145 | 2932  4436 | 107
    8  1  y | 36.6  14.3  22.3 | 1.16  1.84 |   6.65   16.43  | 0.146 0.320 | 3741  5615 | 104
   16  1  y | 61.5  24.3  37.3 | 1.70  2.89 |  16.51   43.10  | 0.313 0.614 | 4445  6650 | 104
   32  1  y | 95.2  37.7  57.5 | 2.15  3.53 |  32.42   81.25  | 0.506 0.818 | 4870  7272 | 103
    8  5  y | 97.8  38.9  58.9 | 2.27  3.81 |  35.26   89.72  | 0.530 0.840 | 4939  7398 | 104
   (prefill=n, same S: m1/step 1.68 / 1.70 / 2.48 / 6.04 / 13.6 / 26.9 / 31.1)

`ws` = distinct pairs a box touched in the last 50 steps; `scan` = distinct box-1
experts per encoder layer per 512-row prefill chunk. What it says:

1. The union grows like ~S^0.75: 16 rows cost 10x one row's expert bytes, 32 cost 16x.
2. Misses per stream-token are flat in S on box 1 (0.6-0.9) and ~2 on box 2; misses
   per STEP therefore grow with S, and the per-lookup rate climbs once the 50-step
   working set exceeds the pool: box 1's reaches 4,445 at S=16 (pool 4,454), box 2's
   6,650 (pool 6,160).
3. REV: prefill scans do not wreck the decode sets. With prefill rows in the pools,
   decode misses per step rise 8-21% at S>=8 (REV2: 6.0 -> 6.7 / 15.2 -> 16.4 at
   S=8; 13.6 -> 16.5 / 37.3 -> 43.1 at S=16), and at S<=2 they FALL (a request's own
   prefill warms its decode set, the unified-pool effect). The scan is only ~104
   box-1 experts per encoder layer per chunk. Caveat: the prefill is replayed as a
   burst at pickup, not chunk-interleaved, and only 1-2 pickups fall in a window.
4. Box 2 takes 2.6x box 1's misses: it owns 60% of the ids at 66.5% residency.
5. REV2: calibration is loose at S=1 and only there. The `prefill=y` S=1 row has
   0.73 box-1 misses/token against 1.2-1.5 live (the trace cycles 9 conversations;
   production runs ~50 agents, so the live LRU is probably ~2x worse than the sim);
   the `prefill=n` row has 1.68. The model uses the `y` rows. These B-invariant
   terms are ~2% of a step at S=16, so no S>=8 conclusion moves.

### 1.3 ESTIMATED parameters (and the measurement that pins each)

* `d1` dGPU chain marginal 8 us/row/layer (M0-a). `a1` per-row attention/indexer/
  compressor 15 us/row/layer at ~100K; the single-row figure is 58 (M0-D4).
  Sensitivity: `a1` 15 -> 58 costs 61 -> 57 tok/s at S=16 in scenario B.
* `bw_moe` 200 GB/s for the by-expert chain on both boxes, cache-defeated (M0-b).
  Sensitivity: 182 -> -4%.
* Engram batched gather 0.6 ms/row exposed in v1, 0 once issued at sample time
  (M0-D1). Host glue 5 ms/step + 0.1 ms/row.
* Fast NVMe 6.5 GB/s effective + 0.5 ms latency; RAID0 13 GB/s (M0-a on the new
  drives). Box 2 per-miss today 4.7 ms all-in (M0-D2; the doc history carries 5.2-8).
* KV per stream 563-895 MB at 307K, 215-330 at 100K (M0-D3).
* REV2: box-2 per-miss cost is the model's most sensitive unmeasured input. At
  6.5 ms all-in instead of 4.7, A2 drops 42.6 -> 36.8 / 50.0 -> 40.2 / 63.8 -> 49.4 at
  S=8/16/32 (-14/-20/-23%). Live data points the other way (live srv - warm srv =
  5.3 ms/token vs the model's 7.9 ms of exposed box-2 reads at S=1). M0-D2 measures
  it and the M2/M3 gates are stated relative to it (section 8).
* REV2: the `+pf` column assumes the M2 chunk gate (>= 500 chunk rows/s with streams
  resident) is met. At 300 rows/s the +pf figures drop ~17% (A2 S=8 29.7 -> 24.7,
  B S=16 37.7 -> 30.0, E S=16 49.3 -> 36.9); 5.3's "~1 s per 256-row tick" is the
  today's-code rate that the chunk miss path in M2 has to lift.
* REV2: the S=1 calibration matches partly by cancellation: miss-free A0 at S=1 is
  51 ms against the measured warm 58.5 (glue 5 vs ~11 measured; the `105 + 20R` fit
  gives srv 18.6 vs 21.6 warm), and the gap is filled by miss terms that are ~2x
  too large on box 2 and ~2x too small on box 1. Irrelevant above S=8.

## 2. The step model (rev 1) and what it says

Per layer: `dense(R) + attn(R) + max(leg_1, leg_2)`; per box `leg = compute +
exposed_read [+ link]`; with hits-first `exposed = P(stall) * (max(0, read -
compute) + launch2)`, with lockstep `exposed = P * read`; `read = lat + k * 18.8 MB /
drive_bw` with `k = E[misses | stalled]` from the simulation. Step = 40 layers +
Engram + glue. Scenarios:

    A0  today's code on today's hardware (lockstep reads, busy-poll hold, Engram per row)
    A1  + hits-first MoE (software)
    A2  + Engram at sample time + multi-segment hold fixed (software)
    B   A2 + 7 GB/s plaintext NVMe in both boxes + concurrent daemon reads
    B2  B + two NVMe per box (RAID0, 13 GB/s)
    C   B + f16 partials + 2nd USB4 link
    B1  A2 + the fast NVMe on BOX 2 ONLY (REV2)
    E   A2 + third 128 GB box: zero misses, hub 29% / remotes 35.5%+35.5% (slot-proportional; REV2)
    F   A2 + IQ2_S requant on 2 boxes: zero misses, 11.06 MB/expert
    S   B + dGPU chain 22 -> 12 ms (software, not built)

DECODE-ONLY aggregate tok/s (per-stream in parentheses), prefill rows in the pools,
E=2.9 for DSpark rows:

    S        A0          A1          A2          B1          B           B2          C           E           F           S
    1     17 (17)     17 (17)     17 (17)     18 (18)     18 (18)     19 (19)     18 (18)     22 (22)     22 (22)     22 (22)
    4     29 (7.4)    31 (7.8)    35 (8.6)    38 (9.4)    38 (9.4)    41 (10.3)   38 (9.6)    54 (14)     54 (14)     41 (10)
    8     35 (4.3)    39 (4.9)    43 (5.3)    48 (6.1)    48 (6.1)    56 (7.0)    50 (6.3)    76 (9.4)    75 (9.4)    52 (6.4)
   16     37 (2.3)    47 (2.9)    50 (3.1)    61 (3.8)    61 (3.8)    72 (4.5)    64 (4.0)    99 (6.2)    99 (6.2)    64 (4.0)
   32     44 (1.4)    60 (1.9)    64 (2.0)    75 (2.4)    82 (2.6)    93 (2.9)    86 (2.7)   128 (4.0)   128 (4.0)    84 (2.6)
   8xK5   29 (3.6)    40 (5.0)    42 (5.3)    51 (6.3)    54 (6.8)    62 (7.8)    58 (7.2)    84 (10.5)   84 (10.5)   55 (6.9)

REV2: the same, WITH the trace's prefill serialized in (5.1 rows per decode token
at 500 chunk rows/s; `aggr+pf` in the model output):

    S        A0          A2          B1          B           B2          E
    4     23 (5.7)    26 (6.4)    27 (6.8)    27 (6.8)    29 (7.3)    35 (8.7)
    8     26 (3.2)    30 (3.7)    32 (4.1)    32 (4.1)    36 (4.5)    43 (5.3)
   16     27 (1.7)    33 (2.1)    37 (2.3)    38 (2.4)    41 (2.6)    49 (3.1)
   32     30 (0.9)    39 (1.2)    43 (1.3)    45 (1.4)    48 (1.5)    55 (1.7)

Prefill alone caps the agent workload at 500 / 5.1 = 98 tok/s whatever decode does,
and is 30-55% of the wall in every scenario; the chunk rate (400-700 tok/s today,
miss-bound on box 1) is therefore a first-order lever of its own (5.3, M2).

Where the S=16 step goes in A0: dense 27, attention 10, leg1 254 (of which exposed
reads 142), leg2 380 (exposed reads 194, link 30), Engram 10 -> 433 ms. In B: 27 +
10 + max(133, 218) + 0 -> 261 ms; in E: 27 + 10 + max(88, 118) -> 161 ms.

Readings:

1. **Hits-first is worth as much as the first hardware step.** A0 -> A1 is +14/+27/
   +37% at S=8/16/32; A2 adds Engram and the hold (+10% at S=8). Both are software.
2. **Fast drives then residency.** B is +14% over A2 at S=8 and +28% at S=32; B2 adds
   +15%. REV2: box 2's drive ALONE (B1) delivers 100 / 99 / 92% of B's gain at
   S=8/16/32, because after the box-2 read shrinks the two legs are nearly equal
   (221 vs 218 ms at S=16) — so item #1 is two purchases with a measurement between
   them. Zero misses (E, F) is +62% over B at S=16 and +56% at S=32, and it is where
   DSpark rows pay clearly (E: 10.5 vs 9.4 per stream at E=2.9; ~16 at 4.4).
3. **Box 2 is the pole at every S** (leg2 > leg1 in every scenario): 60% of the
   union, 2.6x the misses, plus the link. That is what makes the share question real
   (2.4) and why box 2 gets the first fast drive.
4. **The link is second order** (C over B: +4-6%); **the dGPU chain third order** (S
   over B: +8% at S=4, +5% at S=16). Neither gets kernel work in this program.
5. **A0's S=1 (60 ms) sits between the measured warm 58.5 and live 66-78**, with box
   2 exposed by ~8 ms through its misses, matching ROADMAP's "the single-stream bound
   is box 2".

### 2.4 REV: box 1 at 128 GB

The rev-0 claim "changes nothing" assumed misses stay put when the share moves. The
simulation with the actual slot counts says:

    config (hub slots / box-2 slots, hub share)   A1: today's drives     B: fast NVMe both      B2: RAID0 both
                                                   S=8 / 16 / 32 tok/s    S=8 / 16 / 32          S=8 / 16 / 32
    today        4,454 / 6,160, share 0.40          39 / 47 / 60           48 / 61 / 82           56 / 72 / 93
    share only   4,454 / 6,160, share 0.47          29 / 29 / 33  (loss)   52 / 62 / 78           61 / 79 / 102
    SWAP         6,050 / 4,250, share 0.59          33 / 34 / 41  (loss)   52 / 66 / 90           61 / 79 / 103
    swap, 0.47   6,050 / 4,250, share 0.47          27 / 28 / 30  (loss)   37 / 39 / 43           53 / 64 / 75

    (SIMULATED misses per step at S=16, box 1 / box 2: today 16.5 / 43.1; share-only
    55.1 / 14.4 (box 1 residency 62%); swap 42.0 / 28.8 (both ~66%); swap at 0.47
    5.2 / 109.9. The total barely moves; WHERE the misses land does.)

Read: a share change WITHOUT the RAM (today's slots, 0.47) is a loss — box 1's
residency drops to 62% and its misses triple, more than the leg balance recovers.
The swap WITH the share following the slots keeps both residencies at ~66% and
balances the legs, which is worth +7-10% at S=16-32 ONLY once both boxes have fast drives (B: 61 -> 66,
82 -> 90; B2: 72 -> 79, 93 -> 103). With today's drives it is a LOSS of 17-32%: the
swap moves misses onto box 1's dm-crypt YMTC (2.4 GB/s) from box 2's E100 (4.47),
and no leg balance recovers that. It is a second-order lever behind
hits-first and the drives, it costs a day of ops (ROCm stack, repo, OCuLink, both
flake hosts), and its other benefits (no swapping on the hub, room for DSpark's
7.9 GB drafter, Engram page cache) are real but unpriced. Decide after M1 measures
the legs, not before.

## 3. Architecture: one batched step

### 3.1 Rows

A step is `rows: Vec<Row>`, `Row = { stream, token_id, pos, kv: KvRef, kind }`,
`kind in {Decode, Draft(j)}`; rows of a stream are contiguous; R is padded to a
bucket (1, 2, 4, 8, 16, 32, 64, 128); padded rows are masked (no picks, no writes).

### 3.2 Per-stream state

`HetModelState` is one KV state for the engine. Replace with a `KvArena` on the dGPU:
per stream, one region per KV-source layer sized `ctx_now + growth` compressed rows
(grown by realloc+copy), the 40 raw SWA windows and the compressor/index-key state —
ALL on the dGPU, as today. `KvRef` is a per-stream record `{comp_base[4],
index_k_base[4], n_comp[4], n_index_comp[4], raw_base[40], raw_off, n_raw, pos}`
uploaded once per step into a device row table. Every host-side per-token scalar in
the decode layer today (`pos_dev`/`kv_slot_dev`, `ls.n_raw`, the `kv_win` slice at
`raw_off`, the `use_sparse` gate on `n_index_comp`, the compressor boundary predicate
`(pos + 1) % ratio == 0` at forward_layer.rs:962) becomes a row-table read.

### 3.3 Kernels (dGPU dense half)

* **Weight-streaming kernels** (q/kv chains, output proj, shared expert, router, mHC):
  prefill's batched variants for B rows of one sequence do not care which sequence a
  row belongs to. Reuse them, but every one is differenced against its B=1 twin at
  R = 2..64 first: `gemm_batched_wmma` silently zeroes rows below B=64, and nothing
  covers the sub-64 tail of the other tiled kernels (`tests/mtp_batched_kernels_b5.rs`
  is the template).
* **Per-row KV kernels** (`attn_swa`, `attn_mixed` score/smwsum, the indexer chain,
  `kv_append`, the compressor append, `index_k`). REV (rev 1.2, after the launch
  inventory in MULTISTREAM_M1A_INVENTORY.md): batched twins of every one of these
  already exist and are what the prefill/verify driver runs — `attn_swa.launch_batched`
  (`n_raw_per`, `n_raw_offset_per`), `attn_mixed.launch_score_batched_htiled_wmma_f16s`
  and `launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s`, `indexer_score_wmma
  .launch_batched_mw_e2m1` (`n_idx_per`), `indexer_topk_bitonic.launch_batched`,
  `indexer_gather.launch_batched`, `rope.launch_forward_batched` (`pos_per`),
  `kv_append.launch_batched`, the compressor `launch_batched` family (`row_per_b`,
  `pos_mod_per_b`, `comp_pos_per_boundary`), `router_topk.launch_batched`,
  `hc_*`/`rms_*` batched. Their ONE single-sequence assumption is a single KV base
  pointer per store with a uniform per-row stride (`comp_kv_batch_stride`; one
  `raw_kv` base with per-row offsets). So the multi-stream step uses this family
  with per-row BASE OFFSETS into one arena allocation per KV-source layer (the
  KvArena of 3.2), not re-gridded B=1 kernels. Consequences: (a) per-row numerics
  are the prefill/verify family's, so G5b (KLD vs today's decode) is the fidelity
  gate and G5a is self-invariance across buckets and co-rows; (b) the per-row
  causal limit for draft rows is the per-row `n_raw`/`n_comp` those kernels
  already take; (c) what is genuinely new is the per-row base arrays, a batched
  sampler (none exists: `sample_next` is one row with a stream sync and a 4-byte
  D2H), and moving three pieces of single-row engine state per row (`hc_pre_carry`
  is already per lane; `last_idx_gather_src/rows` are engine-level atomics; the
  `engram_rows_ready` flag).
* **REV: the compressor is a per-stream recurrence with a per-row boundary.** For
  K=1 the boundary predicate moves into the kernel (rows at different parities
  fire or no-op individually). For K>1 a stream's draft rows go through it in
  position order: K sequential launches of S rows each (launch j handles every
  stream's j-th row), with a per-row accumulator snapshot after each, which is what
  makes `KvMark` rollback exact at any accepted length (rev 0 claimed exactness the
  batched prefill compressor cannot give). The same K-launch shape applies to
  `kv_append`. Everything else is one launch of S*K rows.
* **Sampler**: `sample_next` is one row with a stream sync and a 4-byte D2H; it
  becomes a row-grid kernel with one D2H per step; multinomial gets a per-row `u01`.
* **Engram**: one batched gather (`rows_for_chunk`) per step for all rows, issued at
  sample time for the NEXT step on a reader pool; 96 random 4 KB reads per row.
  REV2: issue the layer-1 and layer-14 halves SEPARATELY — layer 1 has one layer of
  slack (~6-8 ms at S=16) and its half is 3.4-7.4 ms at S=16 at the measured
  0.42-0.93 ms/row, hidden; at S=32 up to 5 ms may show (<=1.3% of the step).

### 3.4 MoE (both boxes, one kernel family)

REV: rev 0 said "run box 2's `MoeExecutor` on box 1's pool". The executor is the
daemon's HOST-STAGED wrapper (host slices in, pinned staging, a stream sync, D2H of
the partials); on box 1 it would replace two peer pushes and four event waits with
D2H+H2D and 2-3 host syncs per layer, 6-12 ms/step, a 10-20% regression at S=1. What
is shared is the KERNEL CHAIN — `moe_gate_up_chunked` + `launch_by_expert_kwide2` +
`launch_reduce_partials_hetsplit`, Q8_K activations — driven with device-resident
inputs on box 1 over the pager's pool through `layer_views` (whole buffer + slot
remap), with the pager's three-slot-space remap and the `ensure_group_bound`
contract validated against the group builder. Box 2 runs the same chain as today.

The by-expert path has a host readback (`n_work_items` -> grid size) with a stream
sync inside it (remote_experts.rs:2653-2660). Bound the grid at `max_items` with an
in-kernel early exit so the chain becomes capturable and sync-free; do this before
any graph work.

The remote request is unchanged (`submit(layer, b, xq, sel, ew, resp_f32)` carries R
rows; the verify sends B<=8 today). REV2: a third box needs a second
`RemoteExpertClient` (one socket + ordered `in_flight` deque each) so the two remote
legs actually run in parallel; the client is one socket today. Above ~16 rows two link items become real: an f16
reply (`ffn_combine.vec_add_remote` f16 variant, +numerics change, so it lands with
its own KLD gate in M3) and a busy-poll setting that does not hold multi-segment
replies (a 64-row f32 reply is 20 segments; M1). `miss_mask` feedback is b==1-only
and irrelevant under `V41_T2_PARTITION=1`.

### 3.7 REV (rev 1.2): the multi-stream driver is the batched layer driver, parametrized

MULTISTREAM_M1A_DRIVER_MAP.md classifies `forward_layer_pre_moe_v2` (forward_prefill.rs
1787-6167) stage by stage: from the output projection onward — mHC post/pre-ffn,
router, shared expert, local (pager) and remote MoE, combine — it is activation math
over independent rows (the MoE never reads a position or a KV table; the router
reads `tokens[r]` only). The per-sequence surface is exactly: the rope positions
(`bd.pos_per_b`, already a table), the raw window (`ls.kv_cache/n_raw/raw_off`,
the tables derived at FP:2349-2452 and the eviction at FP:4033-4116), the
compressor state (`cs.state_kv/state_score/n_comp/comp_kv/index_k/n_index_comp`,
FP:2613-3053), and the index-k / candidate per-row counts (FP:3500-3505).

So M1a step 3 does NOT write a second driver. It adds a `RowLayout` argument:

    Contiguous { pos0, ls }          today's meaning, byte-identical by construction
    Arena { tables: &RowTables, view: per-layer raw buffer + per-store buffers }

and at the ~15 per-sequence sites uses the table instead of `pos0 + i` /
`ls.n_raw` / `cs.n_comp`: `pos_per_b` from `tables.pos_per`; `n_raw_per`,
`n_raw_offset_per` from the arena (no causal-prefix math: every row is one
position of its own stream); `kv_append.launch_batched_rows(slot_per)`;
the compressor as one-position segments per row (`row_per_b = pos % ratio`,
`compressor_state_write.launch_batched_rows(state_base_per)`, the firing rows via
`compressor_pool.launch_batched_rows(state_idx_per)` -> rms -> index-k -> rope at
`comp_pos_per_boundary` -> fp4 -> `comp_kv_append`/`index_kv_append_*_rows(dst_row_per)`);
`n_comp_per`/`n_index_comp_per_b` from the arena's counters (the "positional
n_comp" audit becomes per row); the attention/indexer launches take
`comp_base_per`/`keys_base_per`; the post-attention eviction becomes
`KvArena::advance` per stream (compaction before the step for streams whose
region is full). Text-only, K=1 rows in v1: no image spans, no MTP capture, no
CED replay rows. The single-sequence `Contiguous` arm keeps every current caller
and the prefill parity tests unchanged; the `Arena` arm is what the harness
exercises. Testing constraint: every driver iteration needs the model loaded, i.e.
the server down — the first harness run rides the M0 window.

**Step 3 WRITTEN 2026-09-20 (untested: needs the model).** What landed:

* `RowLayout<'_>` is the last parameter of `forward_layer_pre_moe_v2`; the five
  existing callers pass `Contiguous`. Under `Arena` the driver takes positions
  from `tables.pos_per` (via one `pos_at(i)` closure that is `pos0 + i` for
  `Contiguous`), the raw counts from `n_raw_per`/`slot_per` (`min(n_raw+1, W)`
  rows ending at the append slot), the append through `kv_append
  .launch_batched_rows(slot_per)`, the compressor as ONE `state_write` over all
  rows into per-stream accumulator blocks (`state_base_per`) followed by the
  pool over the firing rows straight out of their blocks (`fire_state_idx`; no
  snapshot, no shuffle — V4.1 ratios are 1 and 2), the comp / index-K appends at
  `fire_dst_row`, `n_comp_after = n_comp + fires` checked per row against the
  positional formula (own AND reuse layers), the indexer score through
  `launch_batched_mw_e2m1_rows(keys_base_per)` (gemm has no per-row twin; mw is
  bit-exact with it), the gathers through `launch_batched_rows(comp_base_per)`,
  dense attention through the `_f16s_rows` pair with `comp_base_per` (None once
  the indexer gathered), and NO eviction (the arena advances after the step).
  Refused under `Arena`: image visibility, CED modes other than Exact, MTP
  capture, FP8 main stores, `ATTN_FUSED`, f32 scores, `V41_VERIFY_DECODE_ATTN`.
* `KvArena` now holds its buffers inside a `HetModelState` (`arena.state`):
  layer `l`'s `kv_cache` is the arena's raw buffer and the four KV-source layers
  carry a `HetCompressorState` whose accumulators are `[n_slots, ratio*width]`
  and whose `comp_kv`/`index_k` are the shared stores. That is what lets the
  driver take S streams through the same `&mut HetLayerState` (lent by
  `with_kv_source`) it takes one sequence through. `RowTablesDev` holds the
  device copies of the base arrays (`upload` once per step on `de.compute`);
  `admit_from_state` copies a prefilled single-sequence state into a slot (raw
  windows, comp rows, keys, accumulator block, counters).
* `HeterogeneousEngine::forward_step_arena` (forward_prefill.rs): one K=1 step —
  compaction of full regions, tables, `pos_per_b` upload, Engram staging per
  Engram layer, the 40 `with_kv_source(pre_moe(Arena) + post_moe)` calls, advance.
  `head_rows` (now pub) gives the `[b, N_VOCAB]` logits.
* Harness: `crates/v4flash-kernels/tests/multistream_step.rs` (`#[ignore]`): S
  synthetic prompts (lengths 5/40/131/260 by default) prefilled the ordinary way,
  admitted into two arenas; T greedy tokens from today's decode path are the
  forced continuation for `alone` (one row per step) and `batch` (S rows). Reports
  per (stream, step) `|alone - batch|`, argmaxes and KL(dec||batch); G5a = 0
  difference, G5b = KL mean/max under `MS_KLD_MEAN`/`MS_KLD_MAX` (0.02/0.5).
  Run (server DOWN):
  `HIP_VISIBLE_DEVICES=0,1 V41_PAGED_EXPERTS=1 V41_INDEX_K=1 V41_CANDIDATE_POOL=1
  CARGO_TARGET_DIR=target-v41 nix develop -c cargo test -p v4flash-kernels
  --features v41 --release --test multistream_step -- --ignored --nocapture`.
  What the first run tells us: whether the by-expert MoE chain and the batched
  weight-streaming kernels are batch-invariant (G5a, plan 3.6's fallback if not),
  and the arena-vs-decode KL floor (G5b). No remote (box 2) in the harness: all
  experts through the local pager.

**FIRST WINDOW 2026-09-20 (server down 3.5 h; commits 7a5f9ce, 08c6222).**

* **G5a PASSES at S <= 4**: alone vs co-batched bit-identical at every (stream,
  step) in every configuration (Paris + 5/40/131/260-token synthetic prompts, 6
  steps). The by-expert MoE chain IS batch-invariant — 3.6's fallback is not needed.
  At S=8/16 with a 1500-token stream present the indexer fires for the batch and the
  gathered top-K path changes the attention reduction order for every row: alone vs
  batch then differ at the LSB (not bit-exact). G5a at that scale needs a KL-level
  self-invariance metric, not bit equality (harness change pending).
* **G5b: the arena reproduces today's decode at 0.013 nats** (Paris, " Paris")
  when it is the first batched step after one decode token — and is 1.2-2.2 nats
  off after any longer engine history. Root cause open, bisected to the batched
  driver's local MoE output with bit-identical inputs: KNOWN_BUGS #20. Every
  hypothesis tried is listed there. The contiguous verify path has the same class
  of problem in every history (1.14 nats). Until #20 is closed the harness cannot
  pass G5b, and the plan's M1a gate stays open.
* **Decode itself is 0.9-1.2 nats from the CPU oracle** on Paris (KNOWN_BUGS #21,
  pre-existing). The plan's fidelity bar ("decode-vs-oracle 7.x nats" in 3.6) was
  taken from a wrong number and must be restated once #21 is understood.
* **Step time, all experts LOCAL (40 GB pool, no box 2, synthetic prompts,
  including the head):** S=4 108-127 ms, S=8 172-186 ms, S=16 231-251 ms, i.e.
  ~35 / 44 / 65 tok/s aggregate. Not comparable to the model's A-scenario numbers
  (which include box 2's leg and the link); a first data point only.
* Harness reads: the two-prefill KV comparison is bit-identical (prefill is
  deterministic); `forward_prefill_pipelined(last_only=true)` on a continuation
  turns CED on and wipes the decoder layers' window (the reference must pass
  `false`, as the verify does); box 1's NVMe (dm-crypt btrfs) reads 6.3 MB
  O_DIRECT at 3.6 GB/s QD1 (1.7 ms) rising to 4.6 GB/s at QD4-8 — better than the
  2.4 GB/s the model assumed; box 2's fio half did not run (ssh from box 1).

### 3.5 Graphs

Capture per (layer segment, bucket) once the row table carries every per-token
scalar (3.2) and the MoE readback is gone (3.4); graphs are an optimisation after
correctness, not part of M1a. Attention is not captured today and stays that way
until the row-table version is validated.

### 3.6 Correctness gates

* **G5a, bit-exact batch invariance within the new path**: a row's output does not
  depend on its co-rows, its position in the batch, or the bucket (including the
  sub-64-row tile regime). REV2, two references: (i) for the per-row KV kernels
  (`attn_swa`, `attn_mixed`, indexer, `kv_append`, compressor — the same B=1 kernels
  with a row grid) the serial layer-major oracle (`V41_VERIFY_DECODE_PATH=1`,
  forward_layer_standalone_graphs_paged over B rows with per-row `ls`) is bit-exact;
  (ii) for the by-expert MoE chain and the batched weight-streaming kernels the
  oracle runs a different kernel family, so the reference is SELF-invariance: the
  new path at S=1 vs the same row inside S=2..64 buckets and different co-rows.
  Fallback if a kwide kernel's lane/K split turns out to depend on the member count:
  pin the split to a member-count-independent shape (a launch-geometry change, not
  a numerics change), and until then run the local MoE per row with the `hetsplit`
  kernels (slower, invariant by construction). First test written.
* **G5b, KLD against today's decode** under the existing bar (decode-vs-oracle
  7.x nats): the by-expert chain is not bit-identical to the `hetsplit` kernels
  (box 2 measured 1e-7), and 1e-7 flips T=0 tokens over a long generation, so token
  equality is the wrong gate. KLD per token over a 512-token continuation.
* **G6**: single-stream parity tests unchanged.

## 4. Misses under batching

### 4.1 REV: hits-first, both boxes (v1)

Per layer per box: classify the union into resident / missing; launch the resident
groups; issue the missing reads (box 1: `ensure_batched`, already one parallel pass;
box 2: a reader pool with QD 4-8 replacing the serial loop); when the reads land,
launch the missed experts as a second group set into their own `partials` slots
(per (row, slot), written once); reduce in the fixed slot order. Numerics are
unchanged by construction: which experts arrived late does not change what any
partial contains or the order they are summed in. Validate on the daemon first
(M0-D5): it is a ~50-line change in `run_path` and it settles the largest software
unknown in this plan before any kernel is written.

### 4.2 Lockstep (fallback, today's code)

`ensure` then launch. Kept as the reference and as the fallback if hits-first shows
a numeric or scheduling problem on the daemon.

### 4.3 Considered: park a row on its miss and continue

REV: rev 0 priced this with expert misses as row parks (over-counting). Corrected:
at S=16 the 60 misses per step touch ~1.1 rows each, so a row parks ~2-4 times per
token and the batch would still lose most of its rows for most of the step; the
catch-up work re-streams a smaller batch's experts. With hits-first hiding the read
behind compute the motivation is gone. Rejected.

### 4.4 Research: router-ahead prefetch

Run layer L+d's gate matrix on layer L's hidden state (78 MFLOP for all 20 decoder
layers). `V41_PROBE_DUMP` collects the input; break-even precision was 10% for one
stream and is looser under batching (lead time 3-8 ms per layer, 30x the misses).
Offline test from a dump before building anything.

## 5. Scheduler and state machine

### 5.1 Today

`worker_loop` takes one request at a time off an 8-deep FIFO (`try_send` -> 503 when
full) and runs `handle_generate_stream` to completion on the single `state.live`;
`extend`/`exact` have fired 0 times in production; p90 service 210 s of head-of-line
blocking.

### 5.2 Stream lifecycle

    NEW ──admit──> PREFIX ──> PREFILL ──> REPLAY ──> DECODE ──> DONE ──> RESIDENT ──> EVICTED
                     │           ▲                     │                    │
                     │ extend    │ chunks, share-      │ 1 row/step         │ KV kept; next
                     └───────────┘ scheduled, SJF+age  │ (+K draft rows)    │ turn = extend

* **PREFIX**: byte-LCP against every RESIDENT stream first (extend, no disk), then the
  snapshot index (restore into a fresh arena region), else full. With S live states
  the extend path becomes the common case for S concurrent conversations (unpriced
  in the model: restore 7-508 ms + reset 3-132 ms per request today).
* **PREFILL**: encoder-only CED chunks through the existing prefill driver, one prefill
  stream in flight.
* **REPLAY**: the fixed 128-row decoder pass stays on the prefill driver for now (with
  `V41_REPLAY_OFFLOAD`-style decoder residency on box 2, 3.5 s -> 0.2 s). Running it
  as a decode-shaped batch would change the replay's numerics; revisit after G5b.
* **DECODE**: one row per step (+K draft rows with DSpark).
* **DONE -> RESIDENT**: KV stays allocated, snapshotted lazily; LRU-evictable.

### 5.3 REV: step composition

Decode steps and prefill chunks are SEPARATE forwards. Rev 0's v2 "fuse the chunk
with the decode rows' encoder half" is rejected: either the decode rows go through
the prefill chain (their tokens then depend on batch composition, breaking G5a) or
the chunk goes through decode-shaped kernels (losing the 448 -> 727 tok/s prefill
program).

A chunk cannot be bounded by a time budget: at prefill hit 0.88-0.92 a 512-row chunk
pays ~250 box-1 misses (~2.2 s at 8.85 ms; ~1 s at 64 rows, and small chunks are
2-3x less bandwidth-efficient). So the knob is a SHARE: `prefill_share` = the
fraction of scheduler ticks that run a chunk while a prefill is pending (default 1
in 3), and `chunk_rows` (default 256). Decode streams see one step per chunk-tick
of ~1 s during a prefill; the prefill proceeds at ~1/3 speed. With no decode rows
the chunk is `C_max` (512) at full speed. Routing the chunk's MoE to box 2 where its
encoder share is resident (the REPLAY_OFFLOAD mechanism) shortens the chunk and keeps
box 1's pool for decode: when a bug once sent ALL prefill MoE to box 2, prefill went
163 -> 203 tok/s (+25%, 2026-09-13, memory `project_v41_expert_placement_2026-09-13`;
absolute rates were lower then). REV2: because prefill is 30-55% of the agent wall
(section 2), the chunk's miss path — encoder residency on box 2, hits-first for
chunks, the replay offload — is in M2's scope with its own gate.

Prefill queue: shortest remaining suffix first (known after PREFIX), aging bound: a
request is promoted to the head once it has waited 2x its own predicted service
time. Decode admission is FIFO. Tick: gather decode rows -> run step (or chunk) ->
demux samples -> update the live miss-rate estimate.

### 5.4 QoS goals (testable)

    G1  a single stream alone is never slower than today: >= 15 tok/s live, >= 17 warm
    G2  per-stream DECODE floor F under load (REV2: decode-only; G3 bounds the
        prefill share): admit decode streams while the predicted per-stream decode
        rate stays >= F (default 5 tok/s: today's hardware S<=7, fast NVMe S<=10,
        third box S<=20); beyond that, streams queue
    G3  prefill progress: a pending prefill gets >= 1/3 of ticks; TTFT for a 1K-token
        suffix <= 15 s while 8 streams decode (today: unbounded behind a 139K prefill)
    G4  no head-of-line blocking: a 500-token request never waits behind a 139K one
    G5  a/b as in 3.6
    G6  single-stream parity tests unchanged

### 5.5 Admission and memory

* VRAM: ~5.0 GB free today. Per stream 215-330 MB at 100K, 563-895 at 307K (measure).
  Levers, in order: prefill lanes 512 -> 256 rows (~0.4 GB + ~0.26 GB of the
  ctx-scaled shared part); the 32 GB dGPU (section 7); later, host-pinned comp-KV
  values with only the 80 B/row keys and SWA windows on the dGPU — the per-step
  gather is 8 index-source layers x 512 rows x 592 B = 2.4 MB per row-step over
  OCuLink, so it is a real cost (~0.4 ms/row) and an M4+ item.
* Expert pool: REV — the simulation says LRU absorbs the prefill scan (+8-21%
  decode misses at S>=8) because the scan is ~104 experts per encoder layer per
  chunk and decode re-touches its set every step. A decode-protected generation
  (`V41_DECODE_PROTECTED_SLOTS`, capped at ~60% of the pool) is an M2 A/B, scored
  with the miss histogram, not a precondition.
* Host: per-stream host state is small (tokens, Engram id sequence, sampler); the
  Engram gather is batched (3.3).

### 5.6 REV: what a real server needs that rev 0 left out

* **Cancellation** is per row: a cancelled stream leaves the next step, its arena
  region is freed (today `cancel` drops `live`).
* **Failure domain**: a box-2 error or a `verify_routing_exactly_once` failure today
  aborts the request, drains in-flight tickets and reconnects. In a batch that is
  every stream; the drain/reconnect is per STEP, the affected step is retried once
  — REV2: `kv_append` and the compressor advance layer by layer, so a failure at
  layer 25 has appended 25 layers for every row; the step takes a `KvMark` for
  every stream before it starts and `rollback_kv` restores all of them before the
  retry — and per-stream faults (context over `ATTN_MIXED_MAX_KEYS`, bad images)
  are caught at admission so they cannot reach a step.
* **Streaming backpressure**: `tx.blocking_send` from the step loop would stall every
  row on one slow client; use `try_send`, park the stream after N failures, drop it
  after the existing 3 s.
* **Snapshots**: `snapshot::save` is a blocking `copy_to_host` on the worker thread at
  every turn end; with S streams that stalls them all for a restore-sized interval.
  Save on the xfer stream into pinned memory and write from a background thread.
* **Watchdog**: `pet()` per step; the deadline scales with the step's predicted cost.
* **Admission**: a real queue with the G2 floor and a queue-time bound replaces the
  8-deep 503.

## 6. DSpark on the batched path

Rows = streams x (1 + K). The verify IS the batched step over one stream's rows;
fidelity is 3.6 (batch-invariant by construction, KLD-gated against decode, exact
rollback from the K-launch compressor). The drafter (3 blocks, iGPU, 16.8 ms + 11 ms
per accepted token per stream today) must be batched across streams or it
serialises. It pays modestly with fast drives (S=8: 6.8 vs 6.1 tok/s per stream at
E=2.9, ~10 at the oracle E=4.4) and strongly at zero misses. Sequenced after M3.

## 7. Hardware, re-ranked (REV)

| # | item | cost | effect (model) | notes |
|---|---|---|---|---|
| 0 | software first: hits-first MoE, Engram at sample time, multi-segment hold fix | 0 | S=8 35 -> 43, S=16 37 -> 50, S=32 44 -> 64 | M0-D5 validates hits-first on the daemon before anything else |
| 1a | 7 GB/s PCIe 4.0 NVMe with DRAM (990 Pro / SN850X / T500 2 TB) in BOX 2, expert shards on its plaintext /weights | ~$150 | 48 / 61 / 75 at S=8/16/32 = 100 / 99 / 92% of the two-drive gain (REV2) | two empty PCIe slots at 0000:62:00 and 0000:65:00; box 2 takes 2.6x the misses; measure before buying 1b |
| 1b | the same drive in box 1 on a PLAINTEXT partition | ~$150 | 48 / 61 / 82 | box 1's slot table shows only the USB4 tunnels, so a second M.2 needs a physical check; its miss read is under dm-crypt today (2.4 vs 4.3 GB/s ceiling) |
| 2 | second drive per box, RAID0 for the shards | +~$300 | 56 / 72 / 93 | needs the box-2 concurrent reader (4.1) to use the QD |
| 3 | USB4 Gen3 cable + a second USB4 cable | ~$50 | +4-6% above 16 rows (C) | box 1 has a free host router; box 2 exposes one USB4 port in sysfs (`usb4_port2`) — verify a second physical port; the daemon must accept >1 connection |
| 4 | **third 128 GB Strix Halo** (EVO-X2 / Framework Desktop) | ~$1,800-2,000 | zero misses at native precision (16,774 slots > 15,360), three-way split with the hub at 29%: 76 / 99 / 128 decode-only, 43 / 49 / 55 +pf; DSpark pays (84 at S=8xK5, ~130 at E=4.4) | +38% over #2 at S=16/32, +62% over #1; daemon unchanged, client gets a second socket (3.4), hub's second USB4 router carries it |
| 5 | R9700 32 GB (gfx1201) | ~$1,300 | +16 GB VRAM (~50-75 streams at 100K) and a 640 GB/s hot tier of ~750 experts (~20-25% of picks) on a device idle during the MoE phase, ~-15-20% step | after #4; RDNA3 cards are not drop-in (no fp8 WMMA path) |
| 6 | RAM swap (128 GB hub, dGPU moves), share follows slots | a day of ops | -17..-32% with today's drives; +7-10% at S=16-32 after #1 (2.4) | only after #1, and only with the partition share re-tuned; decide after M1 measures the legs |
| alt | IQ2_S requant of the routed experts (170 GB, fits 2 boxes) | software + quality | same ceiling as #4 (F) | V4.1's E2M1 codes are near-uniform (3.9 of 4 bits), so 2.5 bits is lossy; KLD on agent transcripts first |

## 8. Sequencing (REV: M1 split; hits-first and the hold fix moved up; f16 to M3)

**M0 — measure (server down ~30-45 min, ask first).** (a) dGPU chain vs rows: the
prefill driver's per-stage GPU time on 1/4/16/64-token prompts at short context (sets
`d1`; the driver's host overhead is excluded by reading stage GPU time), rocprofv3
kernel trace if the stage brackets are ambiguous. (b) the by-expert MoE chain on box
1's iGPU over the pager's pool at 1..64 rows, D fixed (`--pool 3`) and growing,
cache-defeated: sets `bw_moe` and the 20 us/row term for box 1. (c) DONE 2026-09-21
against a second 3-expert daemon on :7432 (no downtime; the production daemon serves
one connection): any reply over one 65,520-B segment costs ~1.1-1.3 ms of link at
every busy-poll window, ~0.65 ms with TCP_QUICKACK; not autocorking, not a receiver
delayed ACK (kernel counters); RESOLVED the same day: `ethtool -K thunderbolt0 tso off gso
off` on the sender removes the ~1 ms deferral, and the rest scales with the
receiver's busy-poll window (batch phase now 50 us): link 364/404/503/781 us at
4/8/16/32 rows f32 — the model's link term holds (LINK_IDLE_LATENCY.md).
(D1) DONE: the Engram gather is disk-latency-bound (1.7 ms per 24 cold rows on the
loaded drive), a persistent pool changed nothing; the lever is the row cache or the
plaintext drive, and under batching the gather overlaps other rows' work. (d) fio, 6.3 MB reads,
QD 1..16, both drives; on box 1 plaintext and dm-crypt SEPARATELY (needs a plaintext
path; a loop file is not one). (D1) `rows_for_chunk` at 1/4/16/64 rows. (D2) box-2
per-miss cost today from `t_remote_page_us`/`n_remote_miss`. (D3) VRAM of a second
`HetModelState` at 307K and 100K. (D4) `a1`: the serial layer-major driver
(`V41_VERIFY_DECODE_PATH=1`) with 1/4/16 rows at ~100K, per-stage GPU time — the
row cost the row-grid kernels must beat. (D5) DONE 2026-09-21: hits-first is IMPLEMENTED in the daemon (`run_path`: resident
pass, `ensure` with the full pick list while it runs, missed pass into the same
partials, one reduce; `V41_B2_HITS_FIRST=1`, default OFF, `SIGUSR1` flips it at
runtime for a warm-pool A/B). Bit-identical to the single-pass path in 30 checks
(L5/L7, B=1/4/32, 40- and 200-slot paged pools, while faulting). Timing on the
test daemon was NEUTRAL (269 vs 274 ms at B=8, 1045 vs 1175 at B=32, p90 spread
larger than the delta) because a catch-all request there is ~150 SERIAL disk reads
against ~4 ms of compute, i.e. nothing to hide; the predicted gain needs the
production regime (a few misses per layer vs several ms of compute) and is
measured by flipping the flag on the live daemon between token windows. Scripts:
scratchpad `m0_measure.sh` (parts a-d, D).

**M1a — the batched step in a HARNESS (no server changes).** Progress: step 1
DONE 2026-09-21 — per-row KV bases in the seven batched KV kernels (`*_rows`
wrappers; `tests/multistream_row_bases.rs`, bit-identical to single-sequence
runs on the dGPU). Step 2 DONE 2026-09-21 — `het/kv_arena.rs` (regions, tables, advance/compact/restore) and the
row-gridded argmax sampler. Step 3 WRITTEN 2026-09-20 (3.7): `RowLayout` arm in the
batched driver, `KvArena` over a `HetModelState`, `forward_step_arena`, and the
harness `tests/multistream_step.rs` — all compiled, none run (needs the model: the
first run is the M0 window). Row table + KvArena;
row-grid per-row kernels; the K-launch compressor/kv_append; by-expert local MoE with
device inputs; multi-row remote requests; the multi-segment hold fix; row-grid
sampler; G5a against the serial oracle; G5b against today's decode; measured step
time at S = 1, 4, 8, 16 in the harness with the step decomposition. **Gate:** G5a
bit-exact, G5b under the bar, S=8 step <= 230 ms (A0-A1 model 204-232) with the
per-term decomposition matching the model to within 20%.

**M1b — the server.** S live states, scheduler thread, step loop, demux, admission
(VRAM + F), cancellation, failure domain, backpressure, watchdog (5.6); lockstep
misses. **Gate:** S=8 >= 33 tok/s aggregate in a paired A/B against today (the e2e
noise floor is ~8%), G1, G6.

**M2 — scheduler, pool, and the chunk path.** Hits-first on both boxes (validated in
M0-D5); prefill chunks with `prefill_share`; SJF + aging; REPLAY offload; RESIDENT
streams and the extend path; background snapshot writer; the protected-generation
A/B; REV2: the chunk's miss path (encoder residency on box 2, hits-first for chunk
forwards) because prefill is 30-55% of the agent wall. **Gates:** G2-G4 under a
replay of the 50-agent log; decode-only S=16 >= 0.9 x the A2 model re-run with the
M0-D2 box-2 miss cost (45 tok/s at 4.7 ms, 36 at 6.5 ms); chunk rate >= 500 rows/s
with 8 streams resident.

**M3 — miss path + hardware #1-#3.** Concurrent reads on box 2, plaintext fast NVMe
(box 2 first, then measure, then box 1), Gen3/second cable, f16 partials with their
own KLD gate. **Gate:** decode-only S=16 >= 0.9 x the B model at the measured drive
figures (61 tok/s at the assumed 6.5 GB/s); +pf S=16 >= 34.

**M4 — capacity (#4 or the requant).** **Gate:** decode-only S=16 >= 85 tok/s,
per-stream decode >= 5; +pf S=16 >= 44.

**M5 — DSpark on the batched path.** Batched drafter; K=5 rows per stream.
**Gate:** per-stream >= 9 tok/s at S=8.

## 9. Risks and open questions

* `a1` is a guess; if the indexer chain does not batch across rows it is 58
  us/row/layer (-7% at S=16). M0-D4 settles it.
* VRAM at 300K context: 5-9 streams before 5.5's levers. Agents run at 85-221K.
* The multi-segment busy-poll hold (measured today) makes every reply above 3 rows
  f32 slow until fixed; it is in M1a for that reason.
* The daemon is single-connection and single-compute-thread; a second link or a
  prefill lane needs it to serve two sockets, and the lane queueing measured on
  DSpark returns if both share one GPU queue.
* The by-expert chain's host readback and its "correctness depends on the graph
  wrapper" history (remote_experts.rs ~2861) are latent sync hazards for a row-grid
  driver; fixed in 3.4 before capture.
* The simulation's S>8 rows reuse requests at offsets (correlated); the trace has
  only 9 requests. Re-run on a longer multi-agent trace when one exists.
* Box 2's clock skew still bites builds there.

## 10. Considered and rejected

* **Layer-pipeline split**: box 2 would run attention on its iGPU, each box needs all
  experts of its 20 layers (worse residency), KV on both boxes. No.
* **Two-lane ping-pong**: hides the 22 ms chain but halves each lane's union; a wash
  at S=16, a loss at S=32.
* **Fusing prefill chunks with decode rows** (5.3), **parking rows on a miss** (4.3),
  **reusing the daemon's host-staged executor on box 1** (3.4), **page-cache tiers and
  id-based prefetch** (prefetch study).
