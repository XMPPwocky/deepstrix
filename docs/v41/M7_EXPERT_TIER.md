# M7 expert tier — plan (2026-09-13)

V4.1 has 289 GB of MXFP4 experts (384 × 40 × 18.8 MB). Box 1 has 96 GB, box 2 128 GB; even
together (209 GB usable) they cannot hold them all resident. `HetModelWeights::load_all` loads
every expert resident, which OOMs on V4.1. This tier fixes that.

## Reuse
- `WeightSrc::read_expert_into(tensor, e, dst)` (weight_src.rs) → `V41HfWeights::read_expert_into`:
  pread one expert from the mmap'd safetensors + MXFP4 dequant. The per-expert primitive EXISTS.
- M63 hot-expert remap (`HotExpertWeights::load`, `encode_igpu_remap`, `IgpuLayerWeights.hot_remap`):
  loads an expert SUBSET into slots and remaps global→local routing ids. The parity harness already
  drives the MoE kernel over a routed subset this way. The miss path reuses exactly this.

## Phase 1 — correctness-first on-demand (single box, slow, FIRST TOKEN)
Goal: a correct V4.1 token from the Rust engine, e2e vs the CPU oracle. Speed irrelevant.
1. `WorkerState` owns the `V41HfWeights` (mmap must outlive the model). `WeightSrc` is Copy/borrowing
   and cannot be stored beside its owner, so store the owner and build a fresh `WeightSrc::from(&owner)`
   per miss.
2. `load_all` v41 mode (env or cfg): do NOT populate `IgpuLayerWeights` expert slots; allocate an
   empty per-layer expert pool sized to hold one token's routed union (decode ≤ 6+shared; a small cap).
3. Forward hook (decode `forward_token`, prefill `forward_prefill`): before each layer's MoE, read the
   routed expert ids (already computed by the router), pread+dequant them into the pool, build the M63
   remap, run the existing kwide MoE kernel over the pool. Discard/overwrite next layer.
4. Validate: teacher-forced argmax vs `oracle_full`/`oracle_t200` (same tokens the parity harness used),
   then free-running against the oracle's generated tokens (gen_dspark / prose). Gate: argmax match.

## Phase 2 — resident + LRU tier (perf, two boxes)
1. Residency: keep the ~209 GB of hottest experts resident across the two boxes (box 1 iGPU + dGPU hot,
   box 2 iGPU); stream the ~80 GB tail from SSD. Policy = dynamic LRU (COLD_EXPERT_CACHING.md: 11–48×
   better than static). Cache keyed by (layer, expert), capped under the RAM budget, multithreaded pread.
2. Two-box expert-parallel: box 2 runs its resident experts' picks; box 1 sends activations at router
   time, overlaps with local picks (RTT 32 µs measured). This is where 30 tok/s decode comes from.
3. Prefill needs residency, not per-chunk streaming: a 100K prefill at B=1024 touches ~all experts per
   chunk, so streaming would be fatal; CED (20 encoder layers only) + residency is what gets 1000 tok/s.

## Order
Phase 1 first token → measure SSD miss cost and CED prefill rate (the two goal risks) → Phase 2.

## Integration contract — CONFIRMED (2026-09-13, pager built + compiling)
- The iGPU MoE already has the indirection the pager needs: `moe_gate_up_batch_hetsplit` /
  `moe_down_batched_hetsplit` (forward_layer.rs:1627) take `selected` (global ids) + `remap` +
  `mode=0` + `cap`, and index the expert buffer by `slot = -remap[e]-1` for negative entries
  (encode_igpu_remap). MXFP4 arm exists. So NO new kernel: the pager's all-negative remap
  (`-(slot)-1` per routed expert, no dGPU hot set) makes every routed expert compute on the iGPU
  from the packed pool. `ilw.igpu_packed=true` + `ilw.hot_remap=Some(..)` selects this path.
- ExpertPager (het/expert_pager.rs) is built and compiles: owns `V41HfWeights`, a packed
  `RoutedExpertWeights` of `n_slots`, an LRU over `(layer,expert)`; `ensure(layer, ids)` pages
  misses via `read_expert_into`→upload and returns the encoded remap.

## OWNERSHIP DECISION (next step)
The forward reads experts from `ilw.routed` (owned by IgpuLayerWeights) and `ilw.hot_remap`. The
pager owns a persistent pool it reuses across layers, so it cannot be moved into a per-layer
IgpuLayerWeights. Resolution: add a pager-aware forward entry (`forward_layer_standalone_paged` /
thread `routed: &RoutedExpertWeights` + `remap: &DeviceBuffer<i32>` into the MoE dispatch), so the
engine holds ONE ExpertPager (in WorkerState/HetModelState) and passes its pool+remap per layer.
The per-layer norms/router (small) still load per layer or move to a persistent slot. Validate by a
`V41_PAGER=1` layer-major mode: page the dump's routed union per layer, run, confirm residuals match
the resident path; then wire into the server decode/prefill loop.

## PAGER VALIDATED + ENGINE BUG FOUND (2026-09-13)
**The pager is correct in every verifiable respect.** Readback integration built:
`expert_pager.rs` gained a pointer-stable `remap_dev`; `forward_layer.rs` threads
`Option<&mut ExpertPager>` and added `forward_layer_standalone_graphs_paged`, which reads back
`d_selected`, dedupes, `ensure()`s, and runs the het-split MoE from the pool through a
`"routed_moe_paged"` captured graph. Verified by debug readback: it pages the router's ACTUAL picks
(6 distinct ids/token), the LRU reuses slots across tokens, `remap_dev` holds exactly `-(slot)-1`
per pick, and expert bytes are byte-identical to resident (same `read_expert_into` that
`load_experts_packed` uses). MXFP4 het-split mode-0 decode (`e = -remap-1 = slot`) is correct.

**ENGINE BUG (pre-existing, NOT the pager): `hot_experts = None` (all experts on the iGPU) produces
garbage; the dGPU+iGPU split is fine.** Decisive A/B on layer 0:
| config | result |
|---|---|
| resident, forced overflow to iGPU (`DGPU_HOT_CAP=2`, `hot_experts=Some`) | **0.8x floor — correct** |
| graph + resident 384-buffer + raw remap, `hot_experts=None` | **0.96 — garbage** |
The ONLY differing variable is `hot_experts` None vs Some. Not the pool, graph, remap, bytes, or readback.

**Why nothing caught it:** the parity harness sets `DGPU_HOT_CAP = N_EXPERT_USED`, so every resident run
puts ALL experts on the dGPU and the iGPU only ever zeroes. The all-iGPU / no-dGPU-hot MoE path is
exercised by NO test and is broken. NOTE: this may also bite V4-Flash if ever run with hot_k=0.

**Unblock + fix.** The production design keeps a dGPU hot set anyway (M56), so running the pager with
`hot_experts = Some(...)` uses the proven split path and is not a hack. Separately the None path should be
fixed. Next diagnostic to localise: dump `igpu_scratch.ffn_moe` and `ffn_moe_recv` after the MoE for
None vs Some at L0.

## RETRACTION + REAL ROOT CAUSE — PAGER PASSES THE FULL CHAIN (2026-09-13)

**The "`hot_experts=None` engine bug" reported earlier DOES NOT EXIST. Retracted.** That A/B was
confounded: the passing arm never used the pager, so it never touched the mis-allocated buffers.
With the real fix in place, `hot_experts=None` passes with numbers identical to `Some`. Blast
radius: none. No V4-Flash exposure. The final config uses the clean `hot_experts=None`.

**REAL ROOT CAUSE — in the pager, and it is a repo-wide footgun.**
`DeviceBuffer::new(device_id, len)` calls `hipMalloc` **without `hipSetDevice`**: `device_id` is
recorded for bookkeeping/tracing only, so the allocation lands on whatever device is CURRENT.
`ExpertPager::new` never pinned the iGPU, so the expert pool AND `remap_dev` were allocated on the
**dGPU**. The iGPU MoE then read a non-resident remap, got garbage `dense >= 0`, computed
`ours = !dgpu_takes = false`, and wrote `0.0f` into every slot.
**Fix:** `igpu.set_current()` before the allocations in `new`, store the `Device`, pin again in
`ensure()` before the H2D copies.

**RESULT — paged experts reproduce the resident path end to end:**
| chain | per-layer | pager req/miss (hit rate) | HEAD |
|---|---|---|---|
| L0-L7 | 0.7-0.8x floor | 288 / 196 (31.9%) | n/a |
| L0-L21 | 0.5-2.2x floor | 792 / 529 (33.2%) | n/a |
| **L0-L39** | all pass (L37/38/39 same as resident) | 1440 / 1030 (28.5%) | **argmax 11111 = Q8 = fp8 ref; logits 0.68x Q8; PASS** |
Resident regression re-checked: unchanged (0.38x Q8). Ruled out with evidence along the way: graph
capture (`V41_PAGER_NOGRAPH` identical), pool-vs-resident buffer (`V41_PAGER_RESIDENT` gave
BIT-IDENTICAL wrong output), and every kernel scalar.

**=> The M7 core mechanism is VALIDATED: 289 GB of experts can run on a 96 GB box.**

### OPEN HAZARD (route to review): `DeviceBuffer::new`'s `device_id` is inert
It looks like it selects the device but does not; every call site silently depends on ambient
`hipSetDevice` state. This cost a multi-hour misdiagnosis. Either make it call `hipSetDevice` or
rename the argument. It lives in `v4flash-hip`, blast radius = every allocation in the repo.

### Perf notes (not defects, this is the next phase)
- 28.5% hit rate at 48 slots.
- The router readback forces a host sync per layer per token on the decode critical path.
- Paged logits 0.68x Q8 vs resident 0.38x — both well inside the 2x gate; expected, since paged
  computes all routed experts on the iGPU while resident computes most on the dGPU.

## MEASURED: resident footprint and achievable residency (2026-09-13)
Summed from the safetensors shard headers (not estimated):
| bucket | GiB |
|---|---|
| routed experts (PAGED) | 275.7 |
| engram tables (SSD-gathered, never resident) | 189.1 |
| backbone: attn/norms/router/embed/head | 7.6 |
| shared expert | 1.3 |
| vision | 0.9 |
| dspark drafter | 0.7 |
| **TOTAL** | **475.2** |

**Resident with experts paged + Engram SSD-gathered = 10.4 GiB** (8.9 text-only). The server fits
trivially on either box, and vision + drafter add only 1.6 GiB.

**Consequences (this upgrades the plan's assumptions to measurements):**
- Expert size = 275.7 GiB / (384 x 40) = **~18.4 MB**; total experts = 15,360.
- Box 1 (96 GB - 10.4 resident) => ~80 GB pool => **~4,300 slots = 28% residency on one box**,
  versus the 48 slots the pager was validated at (28.5% hit). Hit rate should rise sharply.
- Box 2 (124 GB, remote executor, no backbone needed) => ~120 GB pool => **~6,500 slots = 42%**.
- **Combined ~70% residency**, which MATCHES the ~66% the decode projection in PLAN §7a.1 assumed —
  so the ~26 tok/s single-stream / ~35 with DSpark figures now rest on a measured residency rather
  than a guess. The SSD miss tail is the remaining ~30%.
- Keep real headroom when sizing the pool: the box has OOM'd twice this session.

## CORRECTION (2026-09-13): "batching cannot reach 1000 tok/s" was WRONG
An earlier note claimed the ~289 GB per-chunk expert read meant batching could not reach the prefill
target. That is arithmetically wrong — the per-chunk cost IS amortized by batch size:

    tok/s = B * BW / (L * 384 * 18.8e6 * (1 - r))

with L = layers touched (40 plain, **20 with CED**) and r = resident fraction of the relevant experts.
The bytes are independent of B; the RATE is not.

**The decisive point is CED, not batch heroics.** With CED, prefill touches only the 20 ENCODER layers,
so the relevant expert set is **~144 GB, not 289 GB**. Box 1's ~59 GB pool is ~41% of that; box 1 plus
box 2 (~115 GB there) is ~174 GB, so **the encoder experts fit ENTIRELY resident across the two boxes**.
At full residency the streaming term vanishes and prefill becomes dGPU-compute-bound — which is exactly
where PLAN §7a.1's 1300-1400 tok/s comes from.

Batch size needed for 1000 tok/s at 5 GB/s, with CED (144 GB set):
| residency r | required B |
|---|---|
| 0% | ~28,800 |
| 70% | ~8,600 |
| 90% | ~2,900 |
| ~100% | streaming vanishes; compute-bound |

So on ONE box at ~41% encoder residency the required B is impractical (scratch-bound; B_MAX is 1024
today). The lever is CED + two-box residency, not a bigger batch.

## MEASURED (2026-09-13 ~08:20): residency-aware prefill paging + parallel reads — 7.79 → 14.75 tok/s

1825-token prompt (2 chunks), server settings identical, back-to-back:

| config | time | tok/s | vs base |
|---|---|---|---|
| baseline (r=0, 1 read thread) | 234.2 s | 7.79 | — |
| 4 read threads only | 151.0 s | 12.09 | 1.55× |
| + per-layer dense windows, cold | 137.4 s | 13.29 | 1.71× |
| + per-layer dense windows, warm | 123.7 s | **14.75** | **1.89×** |

- **Why r was 0:** `ensure_layer_dense` wrote layer L's expert e at slot **e**, so all 40 layers collided
  in slots 0..383 (3141 slots, 384 ever used) and every layer evicted the previous one. Fix: per-layer
  384-slot **windows** + a `routed_window()` view so the kernel still indexes by raw expert id (the
  group-builder invariant). Pinning is deterministic (first `windows-1` layers dedicated, the rest rotate):
  a cyclic layer sweep is LRU's worst case (0% hits). Decode's LRU is confined to slots above the dense
  region; a window's identity is cleared BEFORE paging so a mid-failure can't leave it claiming residency
  over a half-written mix of two layers. Effective r = 0.18 (designed 7/40).
- **Parallel reads (attempt #5) WIN** because the precondition in `expert_read_threads`'s comment is now
  met: `read_expert_raw` uses `read_range_into_cached`, so no `POSIX_FADV_DONTNEED` evicts other threads'
  readahead. 2.47 → 3.83 GB/s against the 4.31 GB/s measured ceiling (`V41_PAGER_READ_THREADS=4`).
- Parity unchanged (paged prefill/decode L0..39 identical to resident; HEAD argmax 11111 both modes).
- **Bottleneck unchanged: expert streaming** — 474 GB still moves per warm request. One-box residency is
  structurally capped at `(windows-1)/L` ≈ 18–20%; remaining read headroom ~12%. Next multipliers are
  CED (L 40 → 20, same pool ⇒ r ≈ 0.35) and box 2 (encoder set fully resident).
- Minor: startup log prints "pinned layers 0..7" where it means 0..6.
