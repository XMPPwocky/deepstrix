//! On-demand expert paging for V4.1 (M7 expert tier, phase 1).
//!
//! V4.1 has 289 GB of MXFP4 routed experts (384 × 40 × 18.8 MB) against 96 GB
//! on box 1, so they cannot all be iGPU-resident and `HetModelWeights::load_all`
//! OOMs. This holds the mmap'd HF source plus a fixed pool of `n_slots` expert
//! slots on the iGPU and an LRU over `(layer, expert)`. `ensure(layer, ids)`
//! pages in the misses (one `read_expert_into` per role → upload to a slot) and
//! returns a global→slot remap the MoE kernel consumes exactly like the M63
//! hot-expert remap.
//!
//! Phase 1 is correctness-first: a host readback of the router's picks per layer,
//! then a synchronous page-in of the ≤ TOPK missing experts. Phase 2 (LRU
//! residency across both boxes, overlap, prefetch) is docs/v41/M7_EXPERT_TIER.md.

use std::collections::HashMap;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, V41HfWeights, WeightSrc};
use v4flash_hip::{Device, DeviceBuffer, PinnedBuffer, Stream};

use crate::config::{N_EMBD, N_EXPERT, N_FF_EXP};
use crate::model_weights::RoutedExpertWeights;
use crate::moe_group_builder::MAX_MOE_GROUP_IDS;
use crate::mxfp4_repack::Mxfp4Repack;
use crate::weight_contract;
use crate::weights::DeviceWeight;

/// One role's byte geometry (gate / up / down differ only in k×rows order).
fn role_kr(which: &str) -> (u64, u64) {
    // load_experts_packed: down is [n_ff → n_embd]; gate/up are [n_embd → n_ff].
    if which == "down" {
        (N_FF_EXP as u64, N_EMBD as u64)
    } else {
        (N_EMBD as u64, N_FF_EXP as u64)
    }
}

/// The pool's residency tables: which (layer, expert) each slot holds and how
/// recently each slot was used. Plain host data, kept apart from the HIP
/// buffers in `ExpertPager` so the claim / evict / roll-back rules are
/// unit-testable without a GPU.
struct Residency {
    /// (layer, global expert id) -> resident slot.
    slot_of: HashMap<(i32, u32), u32>,
    /// slot -> the key it currently holds (None = free).
    slot_key: Vec<Option<(i32, u32)>>,
    /// Eviction order as a TICK STAMP per slot, not an ordered list. Was a
    /// `VecDeque<u32>` walked front-to-back, which made every resident HIT
    /// O(pool): `touch` did `iter().position()` plus a mid-deque `remove()`
    /// over up to 4,454 entries. MEASURED 2026-09-22 at 4 rows: `lh.ensure`
    /// cost 59-61 ms/step while the pager missed 0.5 times/step and spent
    /// 4 ms on disk -- ~960 touches/step x a 4,454-entry scan and memmove,
    /// 18% of the step, on the HUB THREAD, for pure bookkeeping. The hit path
    /// is now a single store and the O(pool) work moved into the victim
    /// search, which runs only on a miss. Box 2's pool has always worked this
    /// way (`ShardPool::last_use` + `tick`).
    last_use: Vec<u64>,
    tick: u64,
}

impl Residency {
    fn new(n_slots: usize) -> Self {
        Residency { slot_of: HashMap::new(), slot_key: vec![None; n_slots], last_use: vec![0; n_slots], tick: 0 }
    }

    #[inline]
    fn touch(&mut self, slot: u32) {
        self.tick += 1;
        if let Some(t) = self.last_use.get_mut(slot as usize) {
            *t = self.tick;
        }
    }

    /// Least-recently-used slot in `range`, or `None` if the range is empty.
    /// O(range) and paid only on a MISS; the old deque paid O(pool) per HIT.
    /// A never-used slot has stamp 0 and so is picked first, which is what the
    /// deque did too (it only ever held slots that had been filled).
    fn lru_victim(&self, range: std::ops::Range<usize>) -> Option<u32> {
        let mut best: Option<(u64, u32)> = None;
        for sl in range {
            if sl >= self.last_use.len() {
                break;
            }
            let t = self.last_use[sl];
            if best.is_none_or(|(bt, _)| t < bt) {
                best = Some((t, sl as u32));
            }
        }
        best.map(|(_, sl)| sl)
    }

    /// Give the MISS `key` a slot in `lo..hi` (the first free one, else the LRU
    /// victim, whose key is evicted) and register it at once: `slot_of`,
    /// `slot_key` AND a touch. The touch is what stops the next miss of the
    /// same call from picking this slot again; `ensure_batched` used to claim
    /// without it, so every no-free-slot miss of a layer got the SAME victim,
    /// all of them were uploaded into it (last write wins) and all but one
    /// became permanent hits on another expert's weights.
    ///
    /// `call_tick` is `tick` when the calling `ensure*` started. Every slot the
    /// call has hit or claimed carries a newer stamp, so a victim newer than it
    /// means the range cannot hold the call's set and evicting it would give one
    /// slot to two experts the same dispatch reads: refuse instead.
    ///
    /// The bytes have NOT landed. A caller whose upload fails must `release` the
    /// claim, or the key stays a hit on whatever the slot held before.
    fn claim(&mut self, key: (i32, u32), lo: usize, hi: usize, call_tick: u64) -> eyre::Result<u32> {
        let hi = hi.min(self.slot_key.len());
        let free = (lo..hi).find(|&sl| self.slot_key[sl].is_none());
        let slot = match free {
            Some(sl) => sl as u32,
            None => {
                let victim = self
                    .lru_victim(lo..hi)
                    .ok_or_else(|| eyre!("expert pager: no slot to evict"))?;
                if self.last_use[victim as usize] > call_tick {
                    return Err(eyre!(
                        "expert pager: L{} e{}: every slot in {lo}..{hi} is already used by this \
                         call; the allocation range is smaller than the layer's expert set",
                        key.0,
                        key.1
                    ));
                }
                if let Some(old) = self.slot_key[victim as usize].take() {
                    self.slot_of.remove(&old);
                }
                victim
            }
        };
        self.slot_of.insert(key, slot);
        self.slot_key[slot as usize] = Some(key);
        self.touch(slot);
        Ok(slot)
    }

    /// Undo a `claim` whose bytes never landed: the key stops resolving and the
    /// slot is free again (stamp 0, so it is reused first; its contents are
    /// undefined, possibly half-written).
    fn release(&mut self, key: (i32, u32), slot: u32) {
        if self.slot_of.get(&key) == Some(&slot) {
            self.slot_of.remove(&key);
        }
        if self.slot_key.get(slot as usize) == Some(&Some(key)) {
            self.slot_key[slot as usize] = None;
            self.last_use[slot as usize] = 0;
        }
    }

    /// `release` every claim in `c` past its landed prefix.
    fn release_unlanded(&mut self, layer: i32, c: &Claims) {
        for &(id, slot) in &c.pending[c.landed.min(c.pending.len())..] {
            self.release((layer, id), slot);
        }
    }

    fn clear(&mut self) {
        self.slot_of.clear();
        for k in self.slot_key.iter_mut() {
            *k = None;
        }
        for t in self.last_use.iter_mut() {
            *t = 0;
        }
        self.tick = 0;
    }
}

/// The misses one `ensure*` call claimed, as `(id, slot)` in claim order, and
/// how many of them have landed. Uploads land in claim order, so on an error
/// exactly `pending[landed..]` must be released.
#[derive(Default)]
struct Claims {
    pending: Vec<(u32, u32)>,
    landed: usize,
}

pub struct ExpertPager {
    owner: V41HfWeights,
    /// `n_slots` slots per role; slot `s`'s bytes live at `s * *_bytes_per_expert`.
    pub routed: RoutedExpertWeights,
    n_slots: u32,
    /// Which (layer, expert) each slot holds, and the slots' LRU stamps.
    res: Residency,
    /// `V41_MISS_HIST=1`: misses counted per (layer, expert), so the SHAPE of the
    /// miss stream can be read instead of just its rate. The rate alone cannot
    /// tell apart three cases with opposite fixes — a working set that genuinely
    /// exceeds the pool (capacity), a small set being evicted and re-read
    /// (admission/pinning), and particular layers losing the eviction race
    /// because their reuse distance is longer (eviction policy). `None` unless
    /// the env is set, so this costs nothing in production.
    miss_hist: Option<Vec<std::sync::atomic::AtomicU32>>,
    /// Per-role miss staging in `hipHostMalloc` memory, one buffer per role.
    ///
    /// PINNED, not `Vec<u8>`: this box is an APU, so host-pinned memory IS the
    /// RAM the iGPU reads through GTT. The reader `pread`s straight into it and
    /// the repack kernel consumes it in place, which removes the per-miss H2D
    /// entirely — it was copying system RAM to system RAM. MEASURED on box 1
    /// before this change: `ms_h2d = 1.14` of an 8.24 ms miss, against box 2's
    /// 0.15 ms; box 2 has had pinned staging since its own miss-path rework and
    /// this side never got it back-ported. At ~10,000 box-1 misses per agent
    /// request that is ~11 s of pure copy per request.
    ///
    /// NON_COHERENT so the iGPU may cache its reads (see `PinnedBuffer`).
    stages: Vec<[v4flash_hip::PinnedBuffer<u8>; 3]>,
    /// Scratch remap of length `N_EXPERT`: global id -> slot, or -1 if absent.
    remap: Vec<i32>,
    /// Device copy of `remap`, ONE BUFFER PER LAYER.
    ///
    /// A single shared buffer is only safe while every layer's remap is the
    /// SAME (the dense-window path's identity self-map). The sparse/LRU path
    /// writes per-layer absolute pool slots, so with one buffer, reaching layer
    /// L+1's `ensure` while layer L's MoE is still queued makes layer L's kernel
    /// read L+1's remap -- right expert ids, wrong weights, no error. That race
    /// is why the sparse residency was confined to the single-lane verify and
    /// could not be used by the pipelined prefill driver.
    ///
    /// Per LAYER rather than a ring because each layer's MoE is captured as its
    /// own HIP graph (`igpu_graphs.run("routed_moe_paged", layer, ..)`), which
    /// bakes the buffer pointer: a rotating buffer would have the replay read a
    /// stale one. Box 2 already does exactly this (`LayerShard::remap_dev` +
    /// `pool.dirty[layer]`). 40 x 384 x 4 B = 61 KB.
    remap_dev: Vec<DeviceBuffer<i32>>,
    /// The remap upload is DEFERRED: mutators set this, and `sync_remap_async`
    /// does one stream-ordered async copy before the dispatch reads the buffer.
    ///
    /// It used to be a blocking `hipMemcpy` per mutation (up to three per
    /// lane-layer). MEASURED 2026-09-22: `lh.remap_h2d` = 53-55 ms/step, 17% of
    /// a 318 ms step, to move 123 KB -- because `Stream::new` uses
    /// `hipStreamCreate` (flags 0 = BLOCKING), so a null-stream memcpy waits for
    /// every queued op on BOTH GPUs. Async on the iGPU compute stream orders it
    /// exactly where it is needed (after the previous dispatch that read the old
    /// contents, before the one that reads the new) and returns to the host at once.
    remap_dirty: Option<i32>,
    /// Pinned staging ring for those copies: the host must not rewrite a buffer
    /// while its DMA is pending, and both lanes write the same layer's remap
    /// within a step. 8 x 1536 B. Each slot records an event after its copy and
    /// the host waits on it before rewriting the slot (`remap_stage_done`): the
    /// ring used to rely on the work-item count readback draining the iGPU once
    /// per lane-layer, which `V41_MOE_WI_DEVCOUNT` removed. In practice a lane's
    /// route already waits for its previous MoE (via `selected_ready` after
    /// `moe_arrived`), so at most ~2-5 copies are pending and the wait is free.
    remap_stage: Vec<PinnedBuffer<i32>>,
    remap_stage_done: Vec<Option<v4flash_hip::Event>>,
    remap_stage_next: usize,
    /// Layer whose remap `self.remap` currently describes — the one the next
    /// upload targets. Set by every `ensure*`.
    cur_layer: i32,
    /// Attribute `ensure`'s counters to PREFILL rather than decode.
    ///
    /// *** ON EVERY ARENA DECODE STEP, TOO (2026-09-22 audit A3). *** The
    /// multistream driver sets it around its `ensure` (`forward_prefill.rs`,
    /// "Real prefill: count it as prefill") guarded by `!speculative_append()`,
    /// and `speculative_append()` is only ever true inside DSpark verify, which
    /// is off (`V41_DSPARK=0`). So every decode miss is booked as prefill, which
    /// makes `decode_hit` structurally 1.0000 and `decode_misses` structurally 0
    /// in `het.token.summary` / the request summary. Those two fields are not
    /// measuring anything; read `prefill_misses` or the `ms.stage` rollup instead.
    ///
    /// Gating it on row layout rather than on the verify marker is the fix.
    ///
    /// Under `V41_PREFILL_UNIFIED_POOL` prefill runs through `ensure`, which
    /// counted everything as decode — so `prefill_misses` read 0 while the
    /// prefill's paging hid inside `decode_misses`, making the window path and
    /// the unified path impossible to compare. Set around a prefill `ensure`.
    count_as_prefill: bool,
    /// SCAN REGION: while set, `ensure` may only take slots from `[lo, hi)`.
    ///
    /// Prefill is a scan, not a reuse workload: one 1024-token chunk touches
    /// ~340 distinct experts at each of 40 layers (~13,600 expert-loads against
    /// a 4,454-slot pool), so it can never hit on itself and, run through a
    /// plain LRU, it evicts decode's entire warm working set. Measured cost of
    /// that: warm decode 17.7 tok/s vs cold 3.4.
    ///
    /// The fix is NOT to reserve capacity (prefill and decode never run at the
    /// same time, so reserved windows are dead weight during decode, and the
    /// window path cannot even SEE an expert decode already holds -- it
    /// re-reads it from NVMe at 6-8 ms). It is to split lookup from allocation:
    /// lookup stays global, so prefill hits for free on anything resident;
    /// allocation is confined here, so the scan can only evict itself.
    scan_window: Option<(usize, usize)>,
    /// The iGPU these buffers live on. `DeviceBuffer::new` calls `hipMalloc` on the
    /// CURRENT device and only records `device_id` for bookkeeping, so every alloc and
    /// H2D copy here must pin the device first or the pool/remap silently land on the
    /// dGPU and the iGPU MoE reads a non-resident remap (garbage -> writes zeros).
    device: Device,
    /// Number of dense windows reserved for PREFILL (`dense_windows * N_EXPERT` slots,
    /// from slot 0). Prefill needs slot == expert id inside a window, so one layer
    /// occupies one contiguous 384-slot window; separate windows let DIFFERENT layers
    /// stay resident at once. Before this, every layer was written at slot == id, so
    /// all 40 layers collided in slots 0..384 and each evicted the last — the pool had
    /// 3141 slots but dense mode only ever used 384, giving r_effective = 0.
    dense_windows: u32,
    /// window index -> layer currently resident in it (None = empty/unknown).
    window_layer: Vec<Option<i32>>,
    /// Per window: is EVERY one of the `N_EXPERT` slots filled for `window_layer`?
    /// `ensure_layer_dense` sets it; `ensure_layer_union` clears it, because a
    /// union fill leaves the unrouted slots stale. Without this flag a union fill
    /// followed by a dense request would early-return on `window_layer` alone and
    /// silently compute from another layer's experts.
    window_dense: Vec<bool>,
    /// Slots per PINNED prefill window. `N_EXPERT` (384) reproduces the original
    /// layout exactly; anything smaller PACKS the window.
    ///
    /// A dense window reserved all 384 slots so that `slot == expert id`, but the
    /// dispatch does not require that — `moe_group_builder.hip:116` mode 0 takes
    /// the group id FROM the remap (`g = (dense >= 0) ? e : (-dense - 1)`), and the
    /// MXFP4 kernels index weights by that group id, never by `d_selected`. So a
    /// window only has to be as wide as the union it must hold.
    ///
    /// MEASURED: with box 2 owning 268 of 384 experts per encoder layer, box 1's
    /// unions ran mean 44.1 / max 118 of 384 — 11.5% occupancy. Packing lets the
    /// same pool hold ~3x more windows, which is what fixes the 0.478 prefill hit
    /// rate: with 4 windows over 20 encoder layers a layer's window is always
    /// evicted before the next chunk returns to it.
    ///
    /// Sized from OWNERSHIP, not from that histogram: the cumulative union across
    /// chunks can exceed any single chunk's max, and a different box-2 assignment
    /// enlarges box 1's share. `V41_PAGER_STRIDE` overrides.
    window_stride: u32,
    /// Per-role staging for the batched parallel reader (`batch * bpe` each).
    par_gate: Vec<u8>,
    par_up: Vec<u8>,
    par_down: Vec<u8>,
    /// Paging counters, SPLIT BY PHASE.
    ///
    /// `ensure_layer_dense` is PREFILL's dense-window path (384 requests per
    /// (layer, chunk)); `ensure` is DECODE's LRU path (<= 6 per (layer, token)).
    /// Until 2026-09-13 both fed ONE pair of counters, so every "hit rate" this
    /// project ever quoted was prefill's dense-window residency drowning decode's
    /// 240 requests/token by three orders of magnitude — decode's real miss rate
    /// had never been measured. Never merge these again.
    pub prefill_requests: u64,
    pub prefill_misses: u64,
    pub decode_requests: u64,
    pub decode_misses: u64,
    /// Decode miss-path wall, split by phase: host read (pread + HF->ggml repack)
    /// vs the three synchronous H2D copies.
    pub decode_read_ns: u64,
    pub decode_h2d_ns: u64,
    /// `decode_read_ns` broken down further, taken from the process-wide
    /// `hf_v41::expert_read_profile` around the three `read_expert_into` calls.
    /// Exact for decode because `ensure` runs single-threaded on the engine thread
    /// (prefill's dense path reads on 4 threads, so only its SUM is meaningful).
    pub decode_alloc_ns: u64,
    pub decode_pread_ns: u64,
    pub decode_repack_ns: u64,
    pub decode_pread_bytes: u64,
    /// `decode_pread_ns` split: the O_DIRECT weight read vs the BUFFERED scale
    /// read, so an aggregate that says "slow" can say WHICH read is slow.
    pub decode_weight_ns: u64,
    pub decode_weight_bytes: u64,
    pub decode_scale_ns: u64,
    pub decode_scale_bytes: u64,
    /// Calls per read function, so the byte mismatch localises to a path.
    pub decode_n_layout: u64,
    pub decode_n_runs: u64,
    pub decode_n_direct: u64,
    pub decode_n_raw: u64,
    /// Same for the prefill dense path (batched reads + 3 copies per batch).
    pub prefill_read_ns: u64,
    pub prefill_h2d_ns: u64,
    /// Twins of the decode timers, so `ensure_batched` stops filing PREFILL read
    /// time under `decode_*` while its miss COUNT goes to `prefill_misses`. That
    /// mismatch made `decode_read_ms / decode_misses` mix two phases' numerator
    /// with one phase's denominator and overstated the per-miss cost ~10x.
    ///
    /// **`ensure` was missed by that fix and kept the bug until 2026-09-18.**
    /// It is the path a REAL prefill chunk runs (`forward_prefill.rs`, the
    /// scan-window branch), so `prefill_read_ms` logged 0 on every request that
    /// actually prefilled and prefill's paging cost was never measured at all.
    /// If you are reading a `decode_ms_per_miss` from a per-request line dated
    /// before then, it has prefill's reads in it; the decode HEARTBEAT does
    /// not (it deltas between beats during generation).
    pub prefill_alloc_ns: u64,
    pub prefill_pread_ns: u64,
    pub prefill_repack_ns: u64,
    pub prefill_pread_bytes: u64,
    pub prefill_repack_gpu_ns: u64,
    /// Union-size distribution per `ensure_layer_union` call. Packing a window
    /// to fewer than `N_EXPERT` slots is only safe if the MAX union fits, so
    /// the max — not the mean — is what sizes a packed window.
    pub union_calls: u64,
    pub union_sum: u64,
    pub union_max: u32,
    pub union_min: u32,
    /// Histogram of union sizes in 64-wide buckets (0..64, 64..128, ... 320..384).
    /// A packed window with stride S only works for unions <= S; the tail above S
    /// needs a fallback, so the exceedance rate — not the mean — decides whether a
    /// stride below the worst case is worth the complexity.
    pub union_hist: [u64; 6],
    /// GPU HF->ggml repack (`V41_PAGER_GPU_REPACK`). `None` = CPU repack, the
    /// old behaviour and the only option for a GGUF source.
    ///
    /// The CPU repack measured 1.50 ms of an 11.5 ms decode miss, all of it a
    /// scalar nibble shuffle over 6.3 MB per role. The bytes have to cross the
    /// bus regardless and the HF and ggml forms are the same size, so uploading
    /// the raw form and permuting on the iGPU is free bandwidth we already pay
    /// for. See `kernels/mxfp4_repack.hip`.
    repack: Option<Mxfp4Repack>,
    /// Per-role device landing zone for the raw HF bytes. Three buffers so the
    /// three roles of one miss upload and permute without waiting on each other;
    /// the stream is synchronised once per miss before they are reused.
    repack_scratch: Vec<DeviceBuffer<u8>>,
    /// Box 1 as L1: background reader that fills the decode LRU from this box's
    /// own disk OFF the critical path (`V41_B1_PREFETCH=1`). See [`Prefetcher`].
    prefetch: Option<Prefetcher>,
    readahead: Option<Readahead>,
    repack_stream: Option<Stream>,
    pub decode_repack_gpu_ns: u64,
}

/// Reader threads for the dense prefill window. See the note on
/// `expert_read_threads` in het/weights.rs: four previous attempts at parallel
/// expert reads LOST (42% slower), and the diagnosis was that
/// `POSIX_FADV_DONTNEED` after every read makes concurrent readers evict each
/// other's readahead. That precondition is now gone on THIS path —
/// `read_expert_raw` uses `read_range_into_cached` — so parallelism is worth
/// retesting, but it is measured, not assumed. 1 = serial (old behaviour).
/// `V41_PAGER_BATCH_MISS=0` reverts decode to servicing misses one at a time.
/// On by default: a layer's misses are independent reads and queueing them
/// together is the only way to use the drive's parallel bandwidth.
/// Permute MXFP4 HF->ggml on the iGPU instead of the CPU (`V41_PAGER_GPU_REPACK=0`
/// reverts). Only possible for the HF source; a GGUF is already in ggml layout.
/// Experts box 2 reported as MISSES, per layer — the victim-cache signal.
///
/// Box 1's decode residency used to fill with whatever it happened to see, which
/// is the same hot set box 2's LRU holds. MEASURED consequence (stage diff in
/// `WHY_THE_BIG_POOL_REGRESSED.md`): box 1 cost 140 us/expert to serve picks box 2
/// would have hit for 87 us, and box 2's leg fell only 8 us/pick — a 17x bad
/// trade that made a 1,396-slot decode LRU SLOWER than a 25-slot one.
///
/// The fix is exclusivity: box 1 caches only what box 2 could not serve. Box 2
/// now reports its misses per response (`proto::RESP_MISS_SHIFT`), so box 1 fills
/// from that signal instead of from ambient traffic. Break-even from the same
/// measurement: serving a box-2 HIT is -53 us, serving a box-2 MISS is +6547 us.
///
/// 384 bits per layer, lock-free. Set from the decode wait site, read by the
/// catch-all split. Recency is implicit: an id stays marked until box 1 pages it.
static BOX2_MISSED: std::sync::LazyLock<Vec<std::sync::atomic::AtomicU64>> =
    std::sync::LazyLock::new(|| {
        (0..(crate::config::N_LAYER as usize) * (N_EXPERT as usize).div_ceil(64))
            .map(|_| std::sync::atomic::AtomicU64::new(0))
            .collect()
    });

fn box2_missed_slot(layer: i32, e: u32) -> Option<(usize, u64)> {
    if layer < 0 || layer >= crate::config::N_LAYER as i32 || e >= N_EXPERT {
        return None;
    }
    let per = (N_EXPERT as usize).div_ceil(64);
    Some((layer as usize * per + (e as usize) / 64, 1u64 << (e % 64)))
}

/// Record that box 2 had to page `e` on `layer`.
/// `V41_PICK_TRACE=<path>`: append every router pick to a text trace, one line
/// per (phase, layer, row): `D <layer> <ids...>` for a decode token and
/// `P <layer> <b> <ids...>` for a prefill/verify chunk row. Feeds the offline
/// cache-policy simulator (scratch `simcache.py`), so LRU vs LFU vs two-tier
/// at any capacity is answered from one trace instead of one weight load each.
pub fn pick_trace(line: &str) {
    static W: std::sync::LazyLock<Option<std::sync::Mutex<std::io::BufWriter<std::fs::File>>>> =
        std::sync::LazyLock::new(|| {
            let path = std::env::var("V41_PICK_TRACE").ok()?;
            let f = std::fs::OpenOptions::new().create(true).append(true).open(path).ok()?;
            Some(std::sync::Mutex::new(std::io::BufWriter::new(f)))
        });
    if let Some(m) = W.as_ref() {
        use std::io::Write;
        let mut g = m.lock().unwrap();
        let _ = g.write_all(line.as_bytes());
        let _ = g.write_all(b"\n");
    }
}
pub fn pick_trace_on() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("V41_PICK_TRACE").is_ok());
    *B
}

/// `V41_T2_PARTITION=1`: split the expert id space between the boxes by a
/// fixed hash, box 1's share proportional to its decode-LRU slots vs box 2's
/// pool. Each box then runs its OWN LRU over its OWN demand stream and pages
/// from its OWN disk -- no cross-box protocol, and the simulator says two
/// LRUs over disjoint streams hit like one LRU of the summed capacity
/// (4454+6160: 19.2 -> 5.9 decode misses/token on a 9-request trace).
pub fn t2_partition() -> bool {
    static B: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_T2_PARTITION").as_deref() == Ok("1"));
    *B
}
static PARTITION_BOX1_MILLI: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
/// Set once from (box 1 decode slots, box 2 pool slots). `V41_PARTITION_BOX1_SHARE`
/// (0..1) overrides.
pub fn set_partition_share(box1_slots: u32, box2_slots: u32) {
    let m = std::env::var("V41_PARTITION_BOX1_SHARE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|f| (f.clamp(0.0, 1.0) * 1000.0) as u32)
        .unwrap_or_else(|| (1000 * box1_slots as u64 / (box1_slots as u64 + box2_slots as u64).max(1)) as u32);
    if PARTITION_BOX1_MILLI.swap(m, std::sync::atomic::Ordering::Relaxed) == u32::MAX {
        eprintln!("expert pager: T2 PARTITION on: box 1 takes {:.1}% of expert ids (box1 {box1_slots} slots, box2 {box2_slots})", m as f32 / 10.0);
    }
}
/// Home of `(layer, e)` under the partition: true = box 2. With the HOT-SET
/// ownership warm (`hot_set`), box 1 owns each layer's hottest ids by
/// measured decode pick mass and everything else is box 2's; before that (or
/// with `V41_B1_HOT=0`) the hash split by `V41_PARTITION_BOX1_SHARE`.
pub fn partition_box2(layer: i32, e: u32) -> bool {
    if let Some(own) = hot_set::box1_owns(layer, e) {
        return !own;
    }
    let m = PARTITION_BOX1_MILLI.load(std::sync::atomic::Ordering::Relaxed);
    let m = if m == u32::MAX { 420 } else { m };
    let h = ((layer as u64) * 1000003 + (e as u64) * 7919) % 1000;
    h >= m as u64
}

/// HOT-SET OWNERSHIP (2026-09-22). Box 1's pool (4,150 slots ~ 104/layer) is
/// large enough to hold the hottest ~25% of every layer's experts permanently,
/// and routing is Zipfian, so those ids carry most picks. Owning THEM instead
/// of a random hash slice makes box 1's synchronous miss reads (the step's pole
/// under partitioned paging) nearly vanish, and keeps most of the MoE compute
/// on box 1's otherwise idle iGPU; box 2 becomes the cold tier, whose paging is
/// hidden. The set is measured LIVE (a static placement retains only ~30% of
/// its coverage held out): decode picks are counted per (layer, expert), the
/// per-layer top-K by count (K = `V41_B1_HOT_PER_LAYER`, default 90; also
/// capped at `V41_B1_HOT_MASS` of the layer's mass) is recomputed by the
/// scheduler every `V41_B1_HOT_REFRESH` steps with the counts halved (decaying
/// window). Until the first refresh with enough data the hash split applies.
pub mod hot_set {
    use super::super::hot_split::Side;
    use crate::config::{N_EXPERT, N_LAYER};
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
    const NE: usize = N_EXPERT as usize;
    const NL: usize = N_LAYER as usize;
    static COUNTS: [AtomicU32; NL * NE] = [const { AtomicU32::new(0) }; NL * NE];
    static OWN: [AtomicBool; NL * NE] = [const { AtomicBool::new(false) }; NL * NE];
    static WARM: AtomicBool = AtomicBool::new(false);
    static TOTAL: AtomicU64 = AtomicU64::new(0);
    /// HOT SPLIT (docs/v41/HOT_SPLIT_DESIGN.md 2): each (layer, expert)'s
    /// `hot_split::Side`. `OWN` stays the routing truth for `box1_owns` (= B1
    /// or Rep); STATE adds box 2's head (B2Head: the KEEP set) and the
    /// replicated set. Under `top` it is B1/B2 from `OWN`.
    static STATE: [std::sync::atomic::AtomicU8; NL * NE] = [const { std::sync::atomic::AtomicU8::new(0) }; NL * NE];
    /// Any STATE entry is B2Head or Rep (cheap gate for the KEEP predicate).
    static KEEP_ANY: AtomicBool = AtomicBool::new(false);
    /// The last refresh ran `interleave`.
    static IL_ACTIVE: AtomicBool = AtomicBool::new(false);
    /// Design 2.4: back on `top` after an interleave. Per layer: a returning
    /// layer refreshes with the hysteresis off (change cap on) until box 1 owns
    /// every rank below `per_layer - RETURN_MARGIN` (exact equality never comes:
    /// live counts reshuffle the flat region at the cutoff every refresh).
    static RETURNING: AtomicBool = AtomicBool::new(false);
    static RET_LAYER: [AtomicBool; NL] = [const { AtomicBool::new(false) }; NL];
    static RET_REFRESHES: AtomicU32 = AtomicU32::new(0);
    const RETURN_MARGIN: usize = 10;
    /// Return-mode refreshes before it is forced off (logged).
    const RETURN_MAX_REFRESHES: u32 = 60;
    /// Refresh counter (u32, never 0) and, per (layer, expert), the counter of
    /// the refresh that last moved its HOME between the boxes (0 = never):
    /// design 3.2's MOVING ids, the only ones the holder fallback, the prior's
    /// either-box mask and the ledger's not-stale rule apply to.
    static EPOCH: AtomicU32 = AtomicU32::new(0);
    static MOVED_AT: [AtomicU32; NL * NE] = [const { AtomicU32::new(0) }; NL * NE];
    /// A move counts as MOVING for this many refreshes at most (the
    /// destination usually confirms sooner: `clear_moving`).
    const MOVING_REFRESHES: u32 = 4;
    /// Box-1 newcomers of interleave refreshes, for the paced pre-warm (design 5).
    static PREWARM: std::sync::Mutex<std::collections::VecDeque<(i32, u32)>> =
        std::sync::Mutex::new(std::collections::VecDeque::new());
    /// A refresh changed the KEEP set: the hub re-queues the KEEP pins (design 4).
    static KEEP_DIRTY: AtomicBool = AtomicBool::new(false);
    /// Last refresh's interleave stats (share x1000 mean over layers, swaps,
    /// box-1 newcomers, box-2 newcomers, KEEP size), for the refresh log line.
    static IL_STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

    /// `(layer, e)`'s placement (B2 when cold / out of range).
    #[inline]
    pub fn state(layer: i32, e: u32) -> Side {
        if layer < 0 || layer as usize >= NL || (e as usize) >= NE {
            return Side::B2;
        }
        Side::from_u8(STATE[layer as usize * NE + e as usize].load(Relaxed))
    }
    /// Box 2 must keep `(layer, e)` pinned (design 4: B2Head or Rep). Always
    /// false under `top` (outside an interleave nothing is KEEP).
    #[inline]
    pub fn keep(layer: i32, e: u32) -> bool {
        KEEP_ANY.load(Relaxed) && state(layer, e).keep()
    }
    /// Holder-fallback picks (design 3.2): served by box 1 / by box 2 instead
    /// of their home, since the start.
    static HOLDER_B1: AtomicU64 = AtomicU64::new(0);
    static HOLDER_B2: AtomicU64 = AtomicU64::new(0);
    #[inline]
    pub fn note_holder(on_box2: bool) {
        if on_box2 { &HOLDER_B2 } else { &HOLDER_B1 }.fetch_add(1, Relaxed);
    }
    /// (picks served by box 1 instead of box 2, by box 2 instead of box 1).
    pub fn holder_totals() -> (u64, u64) {
        (HOLDER_B1.load(Relaxed), HOLDER_B2.load(Relaxed))
    }
    /// Is `(layer, e)` MOVING: its home changed boxes within the last
    /// `MOVING_REFRESHES` refreshes and the destination has not confirmed it?
    #[inline]
    pub fn moving(layer: i32, e: u32) -> bool {
        if layer < 0 || layer as usize >= NL || (e as usize) >= NE {
            return false;
        }
        let at = MOVED_AT[layer as usize * NE + e as usize].load(Relaxed);
        at != 0 && EPOCH.load(Relaxed).wrapping_sub(at) < MOVING_REFRESHES
    }
    /// The destination holds it: no longer MOVING.
    #[inline]
    pub fn clear_moving(layer: i32, e: u32) {
        if layer >= 0 && (layer as usize) < NL && (e as usize) < NE {
            MOVED_AT[layer as usize * NE + e as usize].store(0, Relaxed);
        }
    }
    fn next_epoch() -> u32 {
        // u32: one refresh a minute wraps in ~8,000 years; 0 is "never moved".
        let mut v = EPOCH.fetch_add(1, Relaxed).wrapping_add(1);
        if v == 0 {
            v = EPOCH.fetch_add(1, Relaxed).wrapping_add(1);
        }
        v
    }
    /// MOVING ids now (the gate's report).
    pub fn moving_count() -> usize {
        (0..NL * NE).filter(|&i| moving((i / NE) as i32, (i % NE) as u32)).count()
    }
    /// A pre-warm read the prefetcher could not take (in-flight cap): back to
    /// the front of the queue.
    pub fn requeue_prewarm(layer: i32, e: u32) {
        PREWARM.lock().unwrap_or_else(|p| p.into_inner()).push_front((layer, e));
    }
    /// The pin ledger's "box 2 no longer owns it" (stale / lost): today's
    /// `!partition_box2` (the hash split before warm-up included), less the
    /// KEEP set and less a MOVING id box 2 still serves (design 3.2).
    /// Exactly `!partition_box2` under `top` outside a return.
    #[inline]
    pub fn box2_stale(layer: i32, e: u32) -> bool {
        !super::partition_box2(layer, e) && !keep(layer, e) && !moving(layer, e)
    }
    /// Design 3.2 (with review round 2's predicate): should a decode pick of
    /// `(layer, e)` go to box 2? `home_box2` = `partition_box2`. Only a MOVING
    /// id (with `V41_B1_HOT_HOLDER`) leaves its home: box-1 home -> box 2 when
    /// box 1 lacks it and box 2 HOLDS it (pinned: no read, no surprise); box-2
    /// home -> box 1 when box 2 neither holds it nor has it incoming and box 1
    /// has it. The prior's held mask uses the same function, so the mask and
    /// the route agree (a residency change between chain and route costs at
    /// most one read).
    #[inline]
    pub fn serve_on_box2(layer: i32, e: u32, home_box2: bool, b1_has: bool, b2_held: bool, b2_incoming: bool) -> bool {
        if !moving(layer, e) || !crate::knobs::B1_HOT_HOLDER.on() {
            return home_box2;
        }
        if !home_box2 && !b1_has && b2_held {
            return true;
        }
        if home_box2 && !b2_held && !b2_incoming && b1_has {
            return false;
        }
        home_box2
    }
    /// The interleave is in force (its route and ledger rules apply).
    pub fn interleave_active() -> bool {
        IL_ACTIVE.load(Relaxed)
    }
    pub fn returning() -> bool {
        RETURNING.load(Relaxed)
    }
    /// Up to `n` box-1 newcomers to pre-warm.
    pub fn take_prewarm(n: usize) -> Vec<(i32, u32)> {
        let mut g = PREWARM.lock().unwrap_or_else(|p| p.into_inner());
        let k = g.len().min(n);
        g.drain(..k).collect()
    }
    pub fn prewarm_queued() -> usize {
        PREWARM.lock().unwrap_or_else(|p| p.into_inner()).len()
    }
    /// Did a refresh change the KEEP set since the last call?
    pub fn take_keep_dirty() -> bool {
        KEEP_DIRTY.swap(false, Relaxed)
    }
    /// Every KEEP `(layer, e)` (design 4's fill: refresh, decode entry, reconnect).
    pub fn keep_list() -> Vec<(u32, u32)> {
        if !KEEP_ANY.load(Relaxed) {
            return Vec::new();
        }
        let mut v = Vec::new();
        for l in 0..NL {
            for e in 0..NE {
                if Side::from_u8(STATE[l * NE + e].load(Relaxed)).keep() {
                    v.push((l as u32, e as u32));
                }
            }
        }
        v
    }
    /// (box-1 share of the non-replicated mass, swaps, box-1 newcomers,
    /// box-2 newcomers, KEEP size) of the last interleave refresh.
    pub fn il_stats() -> (f32, u64, u64, u64, u64) {
        (IL_STATS[0].load(Relaxed) as f32 / 1000.0, IL_STATS[1].load(Relaxed), IL_STATS[2].load(Relaxed), IL_STATS[3].load(Relaxed), IL_STATS[4].load(Relaxed))
    }
    pub fn enabled() -> bool {
        static B: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var("V41_B1_HOT").as_deref() != Ok("0"));
        *B
    }
    /// Count a decode pick.
    #[inline]
    pub fn note_pick(layer: usize, e: u32) {
        if layer < NL && (e as usize) < NE {
            COUNTS[layer * NE + e as usize].fetch_add(1, Relaxed);
            TOTAL.fetch_add(1, Relaxed);
        }
    }
    /// `Some(owned)` once warm, `None` before.
    #[inline]
    pub fn box1_owns(layer: i32, e: u32) -> Option<bool> {
        if !WARM.load(Relaxed) || layer < 0 || layer as usize >= NL || (e as usize) >= NE {
            return None;
        }
        Some(OWN[layer as usize * NE + e as usize].load(Relaxed))
    }
    pub fn picks_seen() -> u64 { TOTAL.load(Relaxed) }

    /// Decode-LRU slots of box 1's pool, published by `ExpertPager::new`
    /// (0 = no pager yet: the size is not clamped).
    static CAPACITY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    pub fn set_capacity(decode_slots: usize) {
        CAPACITY.store(decode_slots, Relaxed);
    }

    /// The hot set's size per layer that fits `decode_slots` of pool with
    /// `slack` slots per layer to spare (newcomers land before the departed are
    /// evicted).
    pub fn fit_per_layer(want: usize, decode_slots: usize, slack: usize) -> usize {
        if decode_slots == 0 {
            return want;
        }
        want.min((decode_slots / NL).saturating_sub(slack).max(1))
    }

    /// Experts box 1 owns per layer: `V41_B1_HOT_PER_LAYER` (default 90)
    /// CLAMPED to what the pool's decode LRU holds resident, less
    /// `V41_B1_HOT_SLACK` (default 2) per layer. Owning an expert is a routing
    /// promise (box 2 skips it), so an owned expert that does not fit is a
    /// synchronous box-1 read on the critical path each time it is picked. The
    /// size used to be a free knob that nothing re-derived when the pool
    /// shrank: production owned 4,120 in 4,070 decode slots once the 384-slot
    /// prefill band existed, a DSpark hub (7.93 GB drafter) 4,120 in 3,613.
    pub fn per_layer() -> usize {
        let want = env_u("V41_B1_HOT_PER_LAYER", 90).min(NE);
        let cap = CAPACITY.load(Relaxed);
        let fit = fit_per_layer(want, cap, env_u("V41_B1_HOT_SLACK", 2));
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if fit < want && !LOGGED.swap(true, Relaxed) {
            tracing::warn!(want, fit, decode_slots = cap, "box-1 hot set CLAMPED to the pool's decode LRU (V41_B1_HOT_PER_LAYER does not fit)");
        }
        fit
    }
    fn env_f(k: &str, d: f32) -> f32 { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
    fn env_u(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
    /// Recompute the ownership from the counts and decay them. Returns
    /// (owned ids total, mass fraction covered, changed ids) or None when
    /// there is not enough data yet (< `V41_B1_HOT_MIN_PICKS`, default 20000).
    pub fn refresh() -> Option<(usize, f32, usize)> {
        if !enabled() || TOTAL.load(Relaxed) < env_u("V41_B1_HOT_MIN_PICKS", 20_000) as u64 {
            return None;
        }
        if crate::knobs::B1_HOT_POLICY.pick() == 1 && WARM.load(Relaxed) {
            RETURNING.store(false, Relaxed);
            return Some(refresh_interleave());
        }
        // (`interleave` on a cold hot set -- e.g. the knob left on across a
        // restart -- warms up as `top` first: the interleave fills box 1 only at
        // `MOVES` per layer per refresh from an empty start.)
        if IL_ACTIVE.swap(false, Relaxed) {
            // Design 2.4: leaving the interleave. KEEP empties now; every
            // layer returns to the top set at the change cap's pace.
            for r in RET_LAYER.iter() {
                r.store(true, Relaxed);
            }
            RET_REFRESHES.store(0, Relaxed);
            RETURNING.store(true, Relaxed);
        }
        let returning = RETURNING.load(Relaxed);
        let out = refresh_top(returning);
        if returning {
            let left = RET_LAYER.iter().filter(|r| r.load(Relaxed)).count();
            let n = RET_REFRESHES.fetch_add(1, Relaxed) + 1;
            if left == 0 {
                RETURNING.store(false, Relaxed);
                tracing::info!(refreshes = n, "box-1 hot set: RETURN MODE done (every layer owns its top ranks again)");
            } else if n >= RETURN_MAX_REFRESHES {
                for r in RET_LAYER.iter() {
                    r.store(false, Relaxed);
                }
                RETURNING.store(false, Relaxed);
                tracing::warn!(refreshes = n, layers_left = left, "box-1 hot set: RETURN MODE forced off (refresh cap)");
            }
        }
        Some(out)
    }

    /// Today's top-`per_layer` refresh. `returning` (design 2.4): a layer still
    /// returning keeps incumbents only within the top `per_layer` (hysteresis
    /// off); its changes are MOVING and its newcomers are pre-warmed. Outside
    /// a return this is exactly the pre-hot-split refresh.
    fn refresh_top(returning: bool) -> (usize, f32, usize) {
        let per_layer = per_layer();
        let mass_cap = env_f("V41_B1_HOT_MASS", 0.9).clamp(0.0, 1.0);
        // HYSTERESIS (`V41_B1_HOT_HYST`, default 40 ranks): an incumbent keeps
        // its slot until it falls below rank per_layer + hyst; newcomers only
        // fill the slots that frees. Without it a refresh flipped ~1/3 of the
        // set (the cutoff sits in the flat part of the distribution) and every
        // flipped id was a synchronous box-1 read on its next pick.
        let hyst_knob = env_u("V41_B1_HOT_HYST", 40);
        let warm = WARM.load(Relaxed);
        let (mut owned, mut changed, mut mass_sum) = (0usize, 0usize, 0f32);
        let epoch = next_epoch();
        let mut prewarm: Vec<(i32, u32)> = Vec::new();
        for l in 0..NL {
            let ret_l = returning && RET_LAYER[l].load(Relaxed);
            let hyst = if ret_l { 0 } else { hyst_knob };
            let mut v: Vec<(u32, usize)> = (0..NE).map(|e| (COUNTS[l * NE + e].load(Relaxed), e)).collect();
            let total: u64 = v.iter().map(|&(c, _)| c as u64).sum();
            v.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            let mut rank = vec![NE; NE];
            for (i, &(_, e)) in v.iter().enumerate() { rank[e] = i; }
            let mut own = vec![false; NE];
            let mut n_own = 0usize;
            if warm {
                // Incumbents first, if still within the hysteresis band and alive.
                for e in 0..NE {
                    if OWN[l * NE + e].load(Relaxed) && rank[e] < per_layer + hyst && COUNTS[l * NE + e].load(Relaxed) > 0 && n_own < per_layer {
                        own[e] = true;
                        n_own += 1;
                    }
                }
            }
            // Newcomers: top ranks only, into the slots incumbents left free.
            let mut acc = 0u64;
            for (i, &(c, e)) in v.iter().enumerate() {
                if c == 0 || i >= per_layer + hyst { break; }
                if !own[e] && n_own < per_layer && i < per_layer {
                    own[e] = true;
                    n_own += 1;
                }
                if own[e] { acc += c as u64; }
            }
            let _ = mass_cap; // ranks decide; the mass is reported, not enforced
            // CHANGE CAP (`V41_B1_HOT_MAX_CHANGE`, default 0 = unlimited): at
            // most this many NEWCOMERS per layer per refresh; the excess keeps
            // the strongest departing incumbents instead, so the set stays
            // `per_layer` wide. MEASURED 2026-09-27: with hyst 40 a refresh
            // still flipped 255 ids (p90 462) every ~68 s = 208 GB/h of box-1
            // reads while the set's pick mass only moved 0.75-0.83, and the
            // flips are the +18-21 ms of `b1_read` in the slowest decode steps.
            let cap = crate::knobs::B1_HOT_MAX_CHANGE.usize();
            if warm && cap > 0 {
                let prev_own = |e: usize| OWN[l * NE + e].load(Relaxed);
                let mut newcomers: Vec<(u32, usize)> = (0..NE).filter(|&e| own[e] && !prev_own(e)).map(|e| (COUNTS[l * NE + e].load(Relaxed), e)).collect();
                if newcomers.len() > cap {
                    // Weakest newcomers out, strongest departing incumbents back.
                    // Count descending, expert id ascending: deterministic among ties.
                    newcomers.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                    let mut departed: Vec<(u32, usize)> = (0..NE).filter(|&e| !own[e] && prev_own(e)).map(|e| (COUNTS[l * NE + e].load(Relaxed), e)).collect();
                    departed.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                    let excess = newcomers.len() - cap;
                    for &(_, e) in &newcomers[cap..] { own[e] = false; }
                    for &(_, e) in departed.iter().take(excess) { own[e] = true; }
                    // The mass is the accepted set's: recompute it.
                    acc = (0..NE).filter(|&e| own[e]).map(|e| COUNTS[l * NE + e].load(Relaxed) as u64).sum();
                }
            }
            mass_sum += if total > 0 { acc as f32 / total as f32 } else { 0.0 };
            for e in 0..NE {
                let prev = OWN[l * NE + e].swap(own[e], Relaxed);
                if prev != own[e] {
                    changed += 1;
                    if ret_l {
                        MOVED_AT[l * NE + e].store(epoch, Relaxed);
                        if own[e] {
                            prewarm.push((l as i32, e as u32));
                        }
                    }
                }
                if own[e] { owned += 1; }
                STATE[l * NE + e].store(if own[e] { Side::B1 as u8 } else { Side::B2 as u8 }, Relaxed);
                // decay: half-life = one refresh interval
                let c = COUNTS[l * NE + e].load(Relaxed);
                COUNTS[l * NE + e].store(c / 2, Relaxed);
            }
            // Coverage exit: every rank below `per_layer - RETURN_MARGIN` owned.
            if ret_l && (0..NE).all(|e| rank[e] >= per_layer.saturating_sub(RETURN_MARGIN) || own[e]) {
                RET_LAYER[l].store(false, Relaxed);
            }
        }
        if returning {
            push_prewarm(prewarm);
        }
        KEEP_ANY.store(false, Relaxed);
        TOTAL.store(TOTAL.load(Relaxed) / 2, Relaxed);
        WARM.store(true, Relaxed);
        (owned, mass_sum / NL as f32, changed)
    }

    /// Queue pre-warm reads; drop older entries box 1 no longer needs.
    fn push_prewarm(new: Vec<(i32, u32)>) {
        let mut g = PREWARM.lock().unwrap_or_else(|p| p.into_inner());
        g.retain(|&(l, e)| Side::from_u8(STATE[l as usize * NE + e as usize].load(Relaxed)).box1_resident());
        g.extend(new);
    }

    /// Design 2: the sticky interleave (`hot_split::interleave_layer`) per
    /// layer from the decayed counts and the current STATE (today's top set as
    /// B1 on the first interleave refresh). Same decay as `refresh_top`.
    fn refresh_interleave() -> (usize, f32, usize) {
        use crate::knobs as k;
        let first = !IL_ACTIVE.load(Relaxed);
        let mut n1 = k::B1_HOT_IL_PER_LAYER.usize();
        if n1 == 0 {
            n1 = per_layer();
        }
        let n1 = fit_per_layer(n1.min(NE), CAPACITY.load(Relaxed), env_u("V41_B1_HOT_SLACK", 2));
        let p = super::super::hot_split::IlParams {
            n1,
            k: k::B1_HOT_REP.usize(),
            p2: k::B1_HOT_B2HEAD.usize().min(keep_cap_per_layer().saturating_sub(k::B1_HOT_REP.usize())),
            target: k::B1_HOT_TARGET.f64(),
            tol: k::B1_HOT_TOL.f64(),
            moves: k::B1_HOT_MOVES.usize(),
            hyst: k::B1_HOT_IL_HYST.usize(),
            rep_hyst: k::B1_HOT_REP_HYST.usize(),
        };
        let warm = WARM.load(Relaxed);
        let (mut owned, mut changed, mut mass_sum) = (0usize, 0usize, 0f32);
        let (mut share_sum, mut swaps, mut b1n, mut b2n, mut keep_n) = (0f64, 0u64, 0u64, 0u64, 0u64);
        let mut prewarm: Vec<(i32, u32)> = Vec::new();
        let mut counts = vec![0u32; NE];
        let mut prev = vec![Side::B2; NE];
        let epoch = next_epoch();
        for l in 0..NL {
            for e in 0..NE {
                counts[e] = COUNTS[l * NE + e].load(Relaxed);
                prev[e] = if !warm {
                    Side::B2
                } else if first {
                    if OWN[l * NE + e].load(Relaxed) { Side::B1 } else { Side::B2 }
                } else {
                    Side::from_u8(STATE[l * NE + e].load(Relaxed))
                };
            }
            let plan = super::super::hot_split::interleave_layer(&counts, &prev, &p);
            let total: u64 = counts.iter().map(|&c| c as u64).sum();
            let mut acc = 0u64;
            for e in 0..NE {
                let s = plan.sides[e];
                let own = s.box1_resident();
                if OWN[l * NE + e].swap(own, Relaxed) != own {
                    changed += 1;
                }
                if own {
                    owned += 1;
                    acc += counts[e] as u64;
                }
                if s.keep() {
                    keep_n += 1;
                }
                if warm && s.box2_home() != prev[e].box2_home() {
                    MOVED_AT[l * NE + e].store(epoch, Relaxed);
                }
                STATE[l * NE + e].store(s as u8, Relaxed);
                COUNTS[l * NE + e].store(counts[e] / 2, Relaxed);
            }
            mass_sum += if total > 0 { acc as f32 / total as f32 } else { 0.0 };
            share_sum += plan.share1;
            swaps += plan.swaps as u64;
            b1n += plan.b1_new.len() as u64;
            b2n += plan.b2_new.len() as u64;
            if warm {
                prewarm.extend(plan.b1_new.iter().map(|&e| (l as i32, e)));
            }
        }
        push_prewarm(prewarm);
        IL_STATS[0].store((share_sum / NL as f64 * 1000.0) as u64, Relaxed);
        IL_STATS[1].store(swaps, Relaxed);
        IL_STATS[2].store(b1n, Relaxed);
        IL_STATS[3].store(b2n, Relaxed);
        IL_STATS[4].store(keep_n, Relaxed);
        KEEP_ANY.store(keep_n > 0, Relaxed);
        // Every interleave refresh: `keep_fill` dedups against held ids and the
        // queue, so this also re-queues grants that expired unpinned.
        KEEP_DIRTY.store(true, Relaxed);
        IL_ACTIVE.store(true, Relaxed);
        TOTAL.store(TOTAL.load(Relaxed) / 2, Relaxed);
        WARM.store(true, Relaxed);
        (owned, mass_sum / NL as f32, changed)
    }

    /// Design 4's clamp: box 2's KEEP per layer <= (pin budget - 1,024) / 40,
    /// the budget from box 2's replies (`b2_mirror::pin_budget`), else the
    /// production 4,232.
    fn keep_cap_per_layer() -> usize {
        let budget = super::super::b2_mirror::pin_budget().unwrap_or(4232) as usize;
        budget.saturating_sub(1024) / NL
    }

    /// Tests: forget every placement static (they are process-wide).
    #[cfg(test)]
    fn reset_for_test() {
        for i in 0..NL * NE {
            COUNTS[i].store(0, Relaxed);
            OWN[i].store(false, Relaxed);
            STATE[i].store(0, Relaxed);
            MOVED_AT[i].store(0, Relaxed);
        }
        for r in RET_LAYER.iter() {
            r.store(false, Relaxed);
        }
        for f in [&WARM, &KEEP_ANY, &IL_ACTIVE, &RETURNING, &KEEP_DIRTY] {
            f.store(false, Relaxed);
        }
        TOTAL.store(0, Relaxed);
        EPOCH.store(0, Relaxed);
        PREWARM.lock().unwrap().clear();
        use crate::knobs as k;
        for (kn, v) in [(&k::B1_HOT_POLICY, "top"), (&k::B1_HOT_TARGET, "0.60"), (&k::B1_HOT_TOL, "0.02"), (&k::B1_HOT_MOVES, "3"),
                        (&k::B1_HOT_IL_HYST, "40"), (&k::B1_HOT_B2HEAD, "60"), (&k::B1_HOT_IL_PER_LAYER, "0"), (&k::B1_HOT_MAX_CHANGE, "0"),
                        (&k::B1_HOT_HOLDER, "1"), (&k::B1_HOT_REP, "0"), (&k::B1_HOT_REP_HYST, "5")] {
            kn.set(v);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The statics are process-wide: the tests touching them run one at a time.
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

        /// Every layer gets Zipf-like picks over `ne` ids, id order permuted per
        /// layer and perturbed by `noise` (0..1 relative, deterministic LCG).
        fn feed(seed: &mut u64, noise: f64) {
            for l in 0..NL {
                for i in 0..NE {
                    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let r = ((*seed >> 33) as f64) / (1u64 << 31) as f64; // 0..1
                    let base = 2000.0 / (i + 1) as f64;
                    let n = (base * (1.0 + noise * (2.0 * r - 1.0))).max(0.0) as u32;
                    let e = ((i * 7 + l) % NE) as u32;
                    for _ in 0..n {
                        note_pick(l, e);
                    }
                }
            }
        }

        /// HOT SPLIT end to end on the statics: `top` -> `interleave` (share
        /// converges, KEEP and MOVING appear, box-1 newcomers are queued for the
        /// pre-warm, the ledger predicates follow) -> `top` again under noisy
        /// counts (the RETURN MODE ends by coverage within its cap, KEEP is
        /// empty at once, and afterwards `box2_stale == !partition_box2`).
        #[test]
        fn interleave_switch_and_return() {
            let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
            reset_for_test();
            std::env::set_var("V41_B1_HOT_MIN_PICKS", "0");
            std::env::set_var("V41_B1_HOT_PER_LAYER", "40");
            std::env::set_var("V41_B1_HOT_HYST", "20");
            crate::knobs::B1_HOT_MAX_CHANGE.set("3");
            crate::knobs::B1_HOT_B2HEAD.set("20");
            crate::knobs::B1_HOT_TARGET.set("0.35");
            crate::knobs::B1_HOT_MOVES.set("3");
            crate::knobs::B1_HOT_IL_HYST.set("10");
            crate::knobs::B1_HOT_REP.set("0");
            crate::knobs::B1_HOT_POLICY.set("top");
            let mut seed = 1u64;
            for _ in 0..3 {
                feed(&mut seed, 0.0);
                refresh().unwrap();
            }
            assert!(!interleave_active() && !returning());
            assert!(keep_list().is_empty());
            for l in 0..NL as i32 {
                for e in 0..NE as u32 {
                    assert_eq!(box2_stale(l, e), !super::super::partition_box2(l, e));
                }
            }
            // To the interleave.
            crate::knobs::B1_HOT_POLICY.set("interleave");
            let mut share = 1.0;
            for _ in 0..40 {
                feed(&mut seed, 0.0);
                refresh().unwrap();
                share = il_stats().0;
            }
            assert!(interleave_active());
            assert!((share - 0.35).abs() <= 0.03, "share {share}");
            let kl = keep_list();
            assert_eq!(kl.len(), NL * 20, "box 2's head is full");
            assert!(kl.iter().all(|&(l, e)| keep(l as i32, e) && !box2_stale(l as i32, e) && super::super::partition_box2(l as i32, e)));
            assert!(take_keep_dirty());
            assert!(prewarm_queued() > 0, "swapped-in ids queued for the pre-warm");
            // Back to top, counts jittering +-30% every refresh.
            crate::knobs::B1_HOT_POLICY.set("top");
            feed(&mut seed, 0.3);
            refresh().unwrap();
            assert!(returning() && !interleave_active());
            assert!(keep_list().is_empty() && (0..NE as u32).all(|e| !keep(0, e)), "KEEP empties at once");
            let mut n = 1;
            while returning() {
                feed(&mut seed, 0.3);
                refresh().unwrap();
                n += 1;
                assert!(n <= RETURN_MAX_REFRESHES as usize + 1);
            }
            assert!(n > 1 && n < RETURN_MAX_REFRESHES as usize, "ended by coverage after several capped refreshes, not the cap ({n} refreshes)");
            // Moves expire after MOVING_REFRESHES more refreshes.
            for _ in 0..MOVING_REFRESHES {
                feed(&mut seed, 0.0);
                refresh().unwrap();
            }
            for l in 0..NL as i32 {
                for e in 0..NE as u32 {
                    assert!(!moving(l, e));
                    assert_eq!(box2_stale(l, e), !super::super::partition_box2(l, e));
                }
            }
            for k in [&crate::knobs::B1_HOT_MAX_CHANGE] {
                k.set("0");
            }
            reset_for_test();
        }

        /// `interleave` on a COLD hot set (the knob left on across a restart):
        /// the first refresh warms up as `top` (box 1 full at once), the next one
        /// interleaves from that set.
        #[test]
        fn interleave_cold_start_warms_as_top() {
            let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
            reset_for_test();
            std::env::set_var("V41_B1_HOT_MIN_PICKS", "0");
            std::env::set_var("V41_B1_HOT_PER_LAYER", "40");
            crate::knobs::B1_HOT_B2HEAD.set("20");
            crate::knobs::B1_HOT_POLICY.set("interleave");
            let mut seed = 7u64;
            feed(&mut seed, 0.0);
            let (owned, _, _) = refresh().unwrap();
            assert_eq!(owned, NL * 40, "box 1 full after the first refresh");
            assert!(!interleave_active() && keep_list().is_empty());
            feed(&mut seed, 0.0);
            refresh().unwrap();
            assert!(interleave_active());
            assert_eq!(keep_list().len(), NL * 20);
            reset_for_test();
        }

        /// Design 3.2's decision table: only a MOVING id leaves its home, and
        /// only to a box that holds it.
        #[test]
        fn holder_decision() {
            let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
            reset_for_test();
            crate::knobs::B1_HOT_HOLDER.set("1");
            let (l, e) = (3i32, 7u32);
            // Not moving: always home.
            assert!(serve_on_box2(l, e, true, true, false, false));
            assert!(!serve_on_box2(l, e, false, false, true, false));
            let ep = next_epoch();
            MOVED_AT[l as usize * NE + e as usize].store(ep, Relaxed);
            assert!(moving(l, e));
            // Box-1 home, box 1 lacks it, box 2 holds it -> box 2.
            assert!(serve_on_box2(l, e, false, false, true, false));
            // ... but only if box 2 HOLDS it (incoming is not enough).
            assert!(!serve_on_box2(l, e, false, false, false, true));
            // Box-2 home, box 2 neither holds nor has it incoming, box 1 has it -> box 1.
            assert!(!serve_on_box2(l, e, true, true, false, false));
            assert!(serve_on_box2(l, e, true, true, false, true), "incoming: stay on box 2");
            assert!(serve_on_box2(l, e, true, false, false, false), "nobody holds it: home");
            crate::knobs::B1_HOT_HOLDER.set("0");
            assert!(!serve_on_box2(l, e, false, false, true, false), "knob off: home");
            crate::knobs::B1_HOT_HOLDER.set("1");
            clear_moving(l, e);
            assert!(!moving(l, e));
            reset_for_test();
        }

        /// The change cap bounds newcomers per layer per refresh and keeps the
        /// set `per_layer` wide by retaining the strongest departing incumbents.
        /// (One test: the module's counts are process-wide statics.)
        #[test]
        fn refresh_change_cap() {
            let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
            reset_for_test();
            std::env::set_var("V41_B1_HOT_MIN_PICKS", "0");
            std::env::set_var("V41_B1_HOT_PER_LAYER", "4");
            std::env::set_var("V41_B1_HOT_HYST", "0");
            crate::knobs::B1_HOT_MAX_CHANGE.set("1");
            // Warm-up refresh: layer 0 owns 0..4 (counts 40, 30, 20, 10).
            for (e, n) in [(0u32, 40), (1, 30), (2, 20), (3, 10)] {
                for _ in 0..n { note_pick(0, e); }
            }
            let (owned, _, _) = refresh().unwrap();
            assert_eq!(owned, 4);
            assert!((0..4).all(|e| box1_owns(0, e) == Some(true)));
            // Now 10..14 are far hotter than everyone: uncapped, all four would
            // flip; capped at 1 newcomer, only the hottest (10) comes in and the
            // strongest three incumbents stay.
            for (e, n) in [(10u32, 100), (11, 90), (12, 80), (13, 70)] {
                for _ in 0..n { note_pick(0, e); }
            }
            let (owned, _, changed) = refresh().unwrap();
            assert_eq!(owned, 4);
            assert_eq!(changed, 2, "one in, one out");
            assert_eq!(box1_owns(0, 10), Some(true));
            assert!((11..14).all(|e| box1_owns(0, e) == Some(false)));
            assert!((0..3).all(|e| box1_owns(0, e) == Some(true)), "strongest incumbents kept");
            assert_eq!(box1_owns(0, 3), Some(false), "the weakest incumbent left");
            crate::knobs::B1_HOT_MAX_CHANGE.set("0");
            reset_for_test();
        }

        /// The size never exceeds what the decode LRU holds (less slack); an
        /// unknown capacity leaves the knob alone.
        #[test]
        fn hot_set_fits_the_pool() {
            // 2026-10-01 production: 4,070 decode slots, knob 103 -> 99.
            assert_eq!(fit_per_layer(103, 4070, 2), 99);
            // DSpark hub (drafter takes 7.93 GB): 3,613 slots -> 88.
            assert_eq!(fit_per_layer(103, 3613, 2), 88);
            // A knob that fits is kept; no pager = no clamp; never 0.
            assert_eq!(fit_per_layer(80, 4070, 2), 80);
            assert_eq!(fit_per_layer(103, 0, 2), 103);
            assert_eq!(fit_per_layer(103, 40, 2), 1);
        }
    }
}

/// Per-layer count of box-2 DECODE misses (from the `miss_mask` box 2 returns
/// with every decode partial), so a request's misses can be attributed to
/// layers instead of only summed. `take_box2_miss_by_layer` drains it.
pub static BOX2_MISS_BY_LAYER: [std::sync::atomic::AtomicU64; crate::config::N_LAYER as usize] =
    [const { std::sync::atomic::AtomicU64::new(0) }; crate::config::N_LAYER as usize];

pub fn take_box2_miss_by_layer() -> Vec<u64> {
    BOX2_MISS_BY_LAYER.iter().map(|a| a.swap(0, std::sync::atomic::Ordering::Relaxed)).collect()
}

pub fn mark_box2_miss(layer: i32, e: u32) {
    if let Some(c) = BOX2_MISS_BY_LAYER.get(layer as usize) {
        c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if let Some((i, bit)) = box2_missed_slot(layer, e) {
        BOX2_MISSED[i].fetch_or(bit, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Has box 2 missed `e` on `layer` since box 1 last took it?
pub fn box2_missed(layer: i32, e: u32) -> bool {
    box2_missed_slot(layer, e)
        .is_some_and(|(i, bit)| BOX2_MISSED[i].load(std::sync::atomic::Ordering::Relaxed) & bit != 0)
}

/// Clear the mark — call when box 1 has paged it, so the bit means "box 2 missed
/// this and box 1 does not yet hold it".
pub fn clear_box2_miss(layer: i32, e: u32) {
    if let Some((i, bit)) = box2_missed_slot(layer, e) {
        BOX2_MISSED[i].fetch_and(!bit, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Fill box 1's decode LRU only from box-2 misses (`V41_VICTIM_CACHE=0` reverts
/// to filling from ambient traffic, which measured worse — see `BOX2_MISSED`).
pub fn victim_cache() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("V41_VICTIM_CACHE").map(|v| v != "0").unwrap_or(true)
    });
    *B
}

pub fn pager_gpu_repack() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("V41_PAGER_GPU_REPACK").map(|v| v != "0").unwrap_or(true)
    });
    *B
}

pub fn pager_batch_miss() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !matches!(std::env::var("V41_PAGER_BATCH_MISS").as_deref(), Ok("0") | Ok("off"))
    });
    *B
}

fn pager_read_threads() -> usize {
    static N: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("V41_PAGER_READ_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4)
    });
    (*N).max(1)
}

/// `V41_PAGER_COALESCE=1`: read all THREE roles of a miss as one contiguous span
/// (packed nibbles) plus one span of scales, instead of three separate ~6.3 MB
/// reads. MEASURED on this device: 6300k 3,320 MB/s vs 18900k 3,788 MB/s (+14%),
/// and a third as many requests. Falls back automatically when the checkpoint's
/// expert planes are not contiguous (`read_expert_runs_direct` returns None).
///
/// DEFAULT OFF: box 2's equivalent (`V41_B2_COALESCE`) is off because coalescing
/// was once bisected to a CORRUPTION there. `V41_B2_COALESCE_CHECK=1` verifies a
/// coalesced read against the per-role read byte for byte; run the same check
/// here before trusting this.
fn pager_coalesce() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        matches!(std::env::var("V41_PAGER_COALESCE").as_deref(), Ok("1") | Ok("on"))
    });
    *B
}

/// `V41_PAGER_COALESCE_CHECK=1`: after a coalesced read, re-read the SAME expert
/// per-role and compare byte for byte. Mirrors box 2's `V41_B2_COALESCE_CHECK`,
/// which exists because coalescing was once bisected to a CORRUPTION there —
/// a wrong byte here is silent, just worse tokens. Run it with misses frequent
/// (a cold pool) before trusting `V41_PAGER_COALESCE`.
fn pager_coalesce_check() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        matches!(std::env::var("V41_PAGER_COALESCE_CHECK").as_deref(), Ok("1") | Ok("on"))
    });
    *B
}

/// Threads used to service ONE decode miss (`V41_PAGER_MISS_THREADS`). 1 (default) is
/// the original serial gate->up->down read; 3 puts one thread on each role. See the
/// comment at the call site; this is the M8-E floor measurement.
/// `V41_PAGER_MISS_PAR=N`: missing experts of one `ensure` call read concurrently
/// (default 4). MEASURED 2026-09-21 on box 1's btrfs-on-dm-crypt volume with
/// random 18.8 MB O_DIRECT reads: 2.7 GB/s at 1 reader, 3.8 at 4, 4.2 at 8.
fn pager_miss_par() -> usize {
    static N: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("V41_PAGER_MISS_PAR").ok().and_then(|v| v.parse().ok()).unwrap_or(4)
    });
    (*N).clamp(1, 16)
}

fn miss_read_threads() -> usize {
    static N: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("V41_PAGER_MISS_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
    });
    (*N).max(1)
}

/// Experts read per batch before the device copy. Bounds host staging at
/// `batch * 18.8 MB`; 32 keeps it under ~600 MB across the three roles.
/// Should batched prefill page only the chunk's routed union (default) instead
/// of all `N_EXPERT` experts per layer? `V41_PAGER_UNION=0` restores the old
/// dense-always behaviour for A/B.
pub fn pager_union_prefill() -> bool {
    static U: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !matches!(std::env::var("V41_PAGER_UNION").as_deref(), Ok("0") | Ok("off"))
    });
    *U
}

fn pager_read_batch() -> usize {
    static N: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("V41_PAGER_READ_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(32)
    });
    (*N).max(1)
}


/// `V41_B1_PREFETCH=1`: box 1 is the FIRST-level expert cache and box 2 the
/// victim tier. The compute decision stays "resident here -> local, else box 2",
/// but every box-1 miss is queued to a background thread that reads the expert
/// from THIS box's disk and hands the bytes back; `drain_prefetched` admits
/// them into the decode LRU (evicting LRU) at token boundaries, a few per
/// token, so the 8 ms dm-crypt pread never sits on a layer's critical path.
///
/// WHY. Under the T2 catch-all box 1 admitted only into EMPTY slots, and
/// synchronously (8.2 ms blocking each), then froze once full (KNOWN_BUGS
/// #16). So box 2's 6160 slots were the whole dynamic tier (87-91% hit,
/// ~21 misses/token x 6.1 ms) while box 1's slots and drive idled. Fed only by
/// box 1's misses, box 2's LRU converges to what box 1 does NOT hold, so the
/// pair is exclusive without any hint protocol.
/// `V41_B1_PAGE_MISSES=1`: box 1 pages the misses of its partition half itself
/// (batched driver pick loop). Split with `V41_PARTITION_BOX1_SHARE`
/// (bandwidth-proportional: ~0.5 with today's drives).
pub fn b1_page_misses() -> bool {
    static B: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_B1_PAGE_MISSES").as_deref() == Ok("1"));
    *B
}

pub fn b1_prefetch() -> bool {
    static B: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_B1_PREFETCH").as_deref() == Ok("1"));
    *B
}

/// `V41_B1_PREFETCH_ADMIT=N`: admissions per `drain_prefetched` call (default 8;
/// each is ~1.3 ms of H2D + GPU repack on the calling thread).
fn prefetch_admit_per_drain() -> usize {
    static N: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("V41_B1_PREFETCH_ADMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
    });
    *N
}

struct Prefetched {
    layer: i32,
    id: u32,
    /// gate / up / down bytes in the layout `ensure` stages (HF layout when the
    /// GPU repack is on, ggml blocks otherwise). Empty on a read failure.
    bufs: [Vec<u8>; 3],
}

/// Residency changes box 1 made since the last submit, for box 2 (the victim
/// tier) to act on: ADMITTED -> box 2 marks its copy evict-first, so the two
/// pools stay exclusive; EVICTED -> (v2) box 2 re-pages it in the background.
/// Words are `layer << 16 | expert`.
static RESIDENCY_HINTS: std::sync::Mutex<(Vec<u32>, Vec<u32>)> =
    std::sync::Mutex::new((Vec::new(), Vec::new()));

/// Take up to `max` of each kind, oldest first.
pub fn take_residency_hints(max: usize) -> (Vec<u32>, Vec<u32>) {
    let mut g = RESIDENCY_HINTS.lock().unwrap();
    let (na, ne) = (g.0.len().min(max), g.1.len().min(max));
    let a: Vec<u32> = g.0.drain(..na).collect();
    let e: Vec<u32> = g.1.drain(..ne).collect();
    (a, e)
}

fn push_hint(admitted: bool, layer: i32, e: u32) {
    let w = ((layer as u32) << 16) | (e & 0xFFFF);
    let mut g = RESIDENCY_HINTS.lock().unwrap();
    let v = if admitted { &mut g.0 } else { &mut g.1 };
    if v.len() < 4096 {
        v.push(w);
    }
}

/// LIFO request queue: the most recently missed expert is the one most likely
/// to be touched again soon, so it is read first. A FIFO drowned in ~200
/// hints/token and served the oldest.
type HintStack = std::sync::Arc<(std::sync::Mutex<Vec<(i32, u32)>>, std::sync::Condvar)>;

/// `V41_B1_PREFETCH_MIN_TOUCH=N`: queue an expert only once it has missed N
/// times (default 2). A one-touch tail expert costs an admission, a box-2
/// drop (hint) and usually a later re-miss from disk; requiring a repeat keeps
/// the L1 for experts that recur. Counts saturate and are never reset.
fn prefetch_min_touch() -> u8 {
    static N: std::sync::LazyLock<u8> = std::sync::LazyLock::new(|| {
        std::env::var("V41_B1_PREFETCH_MIN_TOUCH").ok().and_then(|v| v.parse().ok()).unwrap_or(2)
    });
    *N
}

struct Prefetcher {
    stack: HintStack,
    rx: std::sync::mpsc::Receiver<Prefetched>,
    /// miss count per (layer, expert), for the min-touch gate.
    touches: std::collections::HashMap<(i32, u32), u8>,
    /// Queued or read-but-not-admitted, so a hint is not queued twice.
    pending: std::collections::HashSet<(i32, u32)>,
    pub queued: u64,
    pub admitted: u64,
    pub dropped_full: u64,
    pub admit_ns: u64,
}

/// Bound on reads in flight or awaiting admission. 64 x 18.8 MB = 1.2 GB of
/// host staging at most.
const PREFETCH_INFLIGHT_MAX: usize = 64;

/// `V41_PREFILL_READAHEAD=1`: while layer L's MoE runs, ask the page cache to
/// pull in the experts layer L+d is about to miss on. Default OFF so it A/Bs in
/// one binary.
///
/// WHY THIS SHAPE, and not a device-side prefetch. Prefill's per-layer `ensure`
/// is synchronous on the host, and its cost is dominated by a BUFFERED pread
/// (`V41_EXPERT_ODIRECT` is off on box 1). A device-side prefetch of layer L+1
/// would have to allocate pool slots and evict, which cannot be done while
/// layer L's MoE kernel may still be queued against those very slots -- the
/// same hazard `V41_SPARSE_REMAP_SYNC` exists for, and silent wrong output if
/// you get it wrong. A page-cache hint touches no slot, no remap and no LRU, so
/// it is safe by construction and its worst case is wasted page cache.
///
/// Only the NON-RESIDENT owned experts are hinted, so the volume is the miss
/// set (prefill hit is ~0.89), not the whole layer. That the union is worth
/// hinting at all is measured: one 512-row chunk touches ~83% of an ENCODER
/// layer's experts, so "everything this layer owns" is a tight over-estimate of
/// "everything this layer will ask for". Decoder layers 20-39 touch 28-40% and
/// are hinted too, just with worse precision. See
/// `docs/v41/PREFETCH_STUDY_2026-09-18.md`.
pub fn prefill_readahead() -> bool {
    static B: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_PREFILL_READAHEAD").as_deref() == Ok("1"));
    *B
}

/// `V41_PREFILL_READAHEAD_DEPTH=N`: how many layers ahead to hint (default 1).
/// Deeper buys more lead time and costs more page cache; box 1 runs a 78 GB
/// pool in 93 GB of RAM, so there is not much of it to spend.
fn prefill_readahead_depth() -> i32 {
    static N: std::sync::LazyLock<i32> = std::sync::LazyLock::new(|| {
        std::env::var("V41_PREFILL_READAHEAD_DEPTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
    });
    (*N).max(1)
}

/// Queue depth for the read-ahead thread. A full queue DROPS the hint rather
/// than blocking prefill: the hint is advisory, and a prefill host thread that
/// waits on a hint has inverted the entire point of this.
const READAHEAD_QUEUE: usize = 4096;

/// fadvise ranges the kernel accepted, cumulative. See `readahead_stats`.
static READAHEAD_RANGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct Readahead {
    tx: std::sync::mpsc::SyncSender<(i32, u32)>,
    queued: u64,
    dropped_full: u64,
}

impl ExpertPager {
    /// Start the prefill read-ahead thread. Like [`Self::start_prefetcher`] it
    /// opens a SECOND handle on the HF dir, so it never touches the pager's own
    /// source and needs no lock against it.
    ///
    /// No-op when `V41_EXPERT_ODIRECT=1`: O_DIRECT bypasses the page cache, so
    /// a `WILLNEED` hint on that data warms something nobody will read.
    pub fn start_readahead(&mut self, dir: &std::path::Path) -> eyre::Result<()> {
        if v4flash_core::hf_v41::expert_odirect() {
            eprintln!(
                "expert pager: prefill read-ahead requested but V41_EXPERT_ODIRECT=1 \
                 bypasses the page cache; NOT started"
            );
            return Ok(());
        }
        let (tx, rx) = std::sync::mpsc::sync_channel::<(i32, u32)>(READAHEAD_QUEUE);
        let dir = dir.to_path_buf();
        std::thread::Builder::new()
            .name("b1-readahead".into())
            .spawn(move || {
                let owner = match V41HfWeights::open(&dir, None) {
                    Ok(o) => o,
                    Err(e) => {
                        eprintln!(
                            "b1-readahead: cannot open {}: {e:#}; read-ahead disabled",
                            dir.display()
                        );
                        return;
                    }
                };
                let src = WeightSrc::from(&owner);
                // `posix_fadvise(WILLNEED)` submits the reads and returns, but it
                // still allocates page-cache pages and can stall under reclaim --
                // which is exactly why it runs here and not on the prefill thread.
                while let Ok((layer, id)) = rx.recv() {
                    for which in ["gate", "up", "down"] {
                        let name = format!("blk.{layer}.ffn_{which}_exps.weight");
                        let Some(t) = src.tensor(&name) else { continue };
                        let n = src.willneed_expert(t, id as usize).unwrap_or(0);
                        READAHEAD_RANGES.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            })?;
        self.readahead = Some(Readahead { tx, queued: 0, dropped_full: 0 });
        eprintln!(
            "expert pager: prefill READ-AHEAD ON (depth {} layer(s), queue {})",
            prefill_readahead_depth(),
            READAHEAD_QUEUE
        );
        Ok(())
    }

    /// Hint every expert of `layer` that this box OWNS and does not already
    /// hold. Called from the prefill loop `depth` layers before the one that
    /// needs them; never blocks (a full queue drops the hint).
    ///
    /// `box2_owns` must be the SAME ownership predicate the layer's own
    /// dispatch will use, or the hint warms bytes the other box will read.
    pub fn readahead_layer(&mut self, layer: i32, box2_owns: impl Fn(u32) -> bool) {
        if layer < 0 || layer >= crate::config::N_LAYER {
            return;
        }
        // Read the residency map BEFORE borrowing `readahead` mutably; only the
        // misses are worth a hint, which is what keeps the volume at the miss
        // set (~11% of the layer) instead of the whole layer.
        let want: Vec<u32> = (0..N_EXPERT)
            .filter(|&e| !box2_owns(e) && !self.is_resident(layer, e))
            .collect();
        let Some(ra) = self.readahead.as_mut() else { return };
        for e in want {
            match ra.tx.try_send((layer, e)) {
                Ok(()) => ra.queued += 1,
                Err(_) => ra.dropped_full += 1,
            }
        }
    }

    /// `(queued, dropped_full, ranges_accepted)` — `None` if read-ahead is off.
    /// `ranges_accepted` counts fadvise calls the kernel took, so a hint that is
    /// silently doing nothing is visible rather than assumed to be working.
    pub fn readahead_stats(&self) -> Option<(u64, u64, u64)> {
        self.readahead.as_ref().map(|r| {
            (
                r.queued,
                r.dropped_full,
                READAHEAD_RANGES.load(std::sync::atomic::Ordering::Relaxed),
            )
        })
    }

    /// How many layers ahead the prefill loop should hint.
    pub fn readahead_depth(&self) -> i32 {
        prefill_readahead_depth()
    }

    /// Start the background reader. Opens a SECOND handle on the HF dir so the
    /// thread never touches the pager's own source.
    pub fn start_prefetcher(&mut self, dir: &std::path::Path) -> eyre::Result<()> {
        let stack: HintStack = std::sync::Arc::new((std::sync::Mutex::new(Vec::new()), std::sync::Condvar::new()));
        let stack_t = stack.clone();
        let (tx_done, rx_done) = std::sync::mpsc::channel::<Prefetched>();
        let sizes = [
            self.routed.gate_bytes_per_expert,
            self.routed.up_bytes_per_expert,
            self.routed.down_bytes_per_expert,
        ];
        let gpu_repack = self.repack.is_some();
        let dir = dir.to_path_buf();
        std::thread::Builder::new()
            .name("b1-prefetch".into())
            .spawn(move || {
                let owner = match V41HfWeights::open(&dir, None) {
                    Ok(o) => o,
                    Err(e) => {
                        eprintln!("b1-prefetch: cannot open {}: {e:#}; prefetch disabled", dir.display());
                        return;
                    }
                };
                let src = WeightSrc::from(&owner);
                loop {
                    let (layer, id) = {
                        let (m, cv) = &*stack_t;
                        let mut q = m.lock().unwrap();
                        while q.is_empty() {
                            q = cv.wait(q).unwrap();
                        }
                        q.pop().unwrap()
                    };
                    let names = [
                        format!("blk.{layer}.ffn_gate_exps.weight"),
                        format!("blk.{layer}.ffn_up_exps.weight"),
                        format!("blk.{layer}.ffn_down_exps.weight"),
                    ];
                    let mut bufs = [vec![0u8; sizes[0]], vec![0u8; sizes[1]], vec![0u8; sizes[2]]];
                    let mut ok = true;
                    for (name, dst) in names.iter().zip(bufs.iter_mut()) {
                        let Some(t) = src.tensor(name) else { ok = false; break };
                        let r = if gpu_repack {
                            src.read_expert_hf_layout(t, id as usize, dst).map(|_| ())
                        } else {
                            src.read_expert_into(t, id as usize, dst)
                        };
                        if let Err(e) = r {
                            eprintln!("b1-prefetch: L{layer} e{id}: {e:#}");
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        bufs = [Vec::new(), Vec::new(), Vec::new()];
                    }
                    if tx_done.send(Prefetched { layer, id, bufs }).is_err() {
                        return;
                    }
                }
            })?;
        self.prefetch = Some(Prefetcher {
            stack,
            rx: rx_done,
            pending: std::collections::HashSet::new(),
            touches: std::collections::HashMap::new(),
            queued: 0,
            admitted: 0,
            dropped_full: 0,
            admit_ns: 0,
        });
        eprintln!(
            "expert pager: box-1 L1 prefetch ON (admit {}/drain, {} in flight max)",
            prefetch_admit_per_drain(),
            PREFETCH_INFLIGHT_MAX
        );
        Ok(())
    }

    /// HOT SPLIT pre-warm (design 5): queue `(layer, id)` for the background read
    /// now -- no min-touch gate (a refresh newcomer is wanted on purpose). Same
    /// dedup and in-flight cap as `prefetch_hint`. `Some(true)` = queued,
    /// `Some(false)` = not needed (no prefetcher, resident, already queued),
    /// `None` = the in-flight cap is full (try again later).
    pub fn prefetch_now(&mut self, layer: i32, id: u32) -> Option<bool> {
        let Some(pf) = self.prefetch.as_mut() else { return Some(false) };
        let key = (layer, id);
        if self.res.slot_of.contains_key(&key) || pf.pending.contains(&key) {
            return Some(false);
        }
        if pf.pending.len() >= PREFETCH_INFLIGHT_MAX {
            return None;
        }
        {
            let (m, cv) = &*pf.stack;
            m.lock().unwrap().push(key);
            cv.notify_one();
        }
        pf.pending.insert(key);
        pf.queued += 1;
        Some(true)
    }

    /// Queue `(layer, id)` for the background read if it is neither resident nor
    /// already queued. Never blocks: a full queue drops the hint (counted).
    pub fn prefetch_hint(&mut self, layer: i32, id: u32) {
        let Some(pf) = self.prefetch.as_mut() else { return };
        let key = (layer, id);
        if self.res.slot_of.contains_key(&key) || pf.pending.contains(&key) {
            return;
        }
        let t = pf.touches.entry(key).or_insert(0);
        *t = t.saturating_add(1);
        if *t < prefetch_min_touch() {
            return;
        }
        if pf.pending.len() >= PREFETCH_INFLIGHT_MAX {
            pf.dropped_full += 1;
            return;
        }
        {
            let (m, cv) = &*pf.stack;
            m.lock().unwrap().push(key);
            cv.notify_one();
        }
        pf.pending.insert(key);
        pf.queued += 1;
    }

    /// Admit up to `V41_B1_PREFETCH_ADMIT` completed reads into the decode LRU
    /// (free slot above the dense region, else the LRU victim), same slot rules
    /// and upload path as `ensure`'s miss branch. Call ONLY at a point where no
    /// MoE kernel can still be reading the pool (a token / verify-step boundary):
    /// the victim slot is overwritten in place. `remap` is untouched -- `ensure`
    /// rewrites it per layer from `slot_of`.
    pub fn drain_prefetched(&mut self) -> eyre::Result<usize> {
        let Some(pf) = self.prefetch.as_mut() else { return Ok(0) };
        let cap = prefetch_admit_per_drain();
        let mut got: Vec<Prefetched> = Vec::new();
        while got.len() < cap {
            match pf.rx.try_recv() {
                Ok(p) => got.push(p),
                Err(_) => break,
            }
        }
        if got.is_empty() {
            return Ok(0);
        }
        let t0 = std::time::Instant::now();
        // Scoped, NOT `set_current`: the caller (het engine) mirrors the current
        // device in a cache, and a bare switch here leaves its next cached switch
        // a no-op on the wrong device -> hipErrorInvalidHandle on the next launch.
        let _dev = self.device.scoped_current()?;
        let mut n = 0usize;
        for p in got {
            let key = (p.layer, p.id);
            if let Some(pf) = self.prefetch.as_mut() {
                pf.pending.remove(&key);
            }
            if p.bufs[0].is_empty() || self.res.slot_of.contains_key(&key) {
                continue; // read failed, or `ensure` paged it meanwhile
            }
            let lru_lo = {
                let lo = self.dense_slots();
                if lo >= self.n_slots as usize { 0 } else { lo }
            };
            let slot = match self.res.slot_key.iter().enumerate().skip(lru_lo).find(|(_, k)| k.is_none()) {
                Some((free, _)) => free as u32,
                None => {
                    let victim = self
                        .res
                        .lru_victim(lru_lo..self.n_slots as usize)
                        .ok_or_else(|| eyre!("expert pager: prefetch has no slot to evict"))?;
                    if let Some(old) = self.res.slot_key[victim as usize].take() {
                        self.res.slot_of.remove(&old);
                        push_hint(false, old.0, old.1);
                    }
                    victim
                }
            };
            if (slot as usize) < self.dense_slots() {
                if let Some(w) = self.window_of_slot(slot) {
                    if let Some(e) = self.window_layer.get_mut(w as usize) { *e = None; }
                    if let Some(d) = self.window_dense.get_mut(w as usize) { *d = false; }
                }
            }
            // HOT SPLIT: box 2 must keep a KEEP id (design 4) -- no "box 1 has
            // it" demotion; and a box-1-home id that lands is done MOVING.
            if !hot_set::keep(p.layer, p.id) {
                push_hint(true, p.layer, p.id);
            }
            if !partition_box2(p.layer, p.id) {
                hot_set::clear_moving(p.layer, p.id);
            }
            if self.repack.is_some() {
                let (rp, st) = (self.repack.as_ref().unwrap(), self.repack_stream.as_ref().unwrap());
                Self::upload_and_repack(
                    rp, st, &mut self.repack_scratch, &mut self.routed, slot,
                    [&p.bufs[0], &p.bufs[1], &p.bufs[2]],
                )?;
            } else {
                let gbpe = self.routed.gate_bytes_per_expert;
                let ubpe = self.routed.up_bytes_per_expert;
                let dbpe = self.routed.down_bytes_per_expert;
                self.routed.gate.buffer.slice_view_mut(slot as usize * gbpe, gbpe).copy_from_host(&p.bufs[0])?;
                self.routed.up.buffer.slice_view_mut(slot as usize * ubpe, ubpe).copy_from_host(&p.bufs[1])?;
                self.routed.down.buffer.slice_view_mut(slot as usize * dbpe, dbpe).copy_from_host(&p.bufs[2])?;
            }
            assert!(slot < self.n_slots, "expert pager: prefetch slot {slot} >= pool {}", self.n_slots);
            self.res.slot_of.insert(key, slot);
            self.res.slot_key[slot as usize] = Some(key);
            self.res.touch(slot);
            clear_box2_miss(p.layer, p.id);
            n += 1;
        }
        if let Some(pf) = self.prefetch.as_mut() {
            pf.admitted += n as u64;
            pf.admit_ns += t0.elapsed().as_nanos() as u64;
        }
        Ok(n)
    }

    /// (queued, admitted, dropped_full, admit_ms) since start; None if off.
    pub fn prefetch_stats(&self) -> Option<(u64, u64, u64, u64)> {
        self.prefetch.as_ref().map(|p| (p.queued, p.admitted, p.dropped_full, p.admit_ns / 1_000_000))
    }
}

impl ExpertPager {
    /// Allocate `n_slots` expert slots on `igpu` and take ownership of the HF
    /// source (its mmap must outlive the model). `n_slots` must be at least the
    /// per-token routed union the caller will ask for at once (decode: ≤ TOPK).
    ///
    /// `n_slots == 0` auto-sizes from a RAM budget: with experts paged and the
    /// Engram tables SSD-gathered, the resident model is only ~10 GiB, so most of
    /// the box is free for pool. `V41_PAGER_SLOTS` overrides the count outright;
    /// otherwise `V41_PAGER_POOL_GB` (default 60) picks it. Pool size affects only
    /// the hit rate, never correctness — the LRU is size-agnostic — so it is safe
    /// to tune. Headroom matters: this box has OOM'd on over-allocation before.
    pub fn new(owner: V41HfWeights, igpu: Device, n_slots: u32) -> eyre::Result<Self> {
        let device_id = igpu.id;
        // MUST pin the device: DeviceBuffer::new hipMallocs on the CURRENT device.
        igpu.set_current()?;
        let src = WeightSrc::from(&owner);
        // Geometry first (no allocation): the per-expert byte cost decides how many
        // slots the RAM budget buys.
        let geom = |which: &str| -> eyre::Result<(GgufType, u64, u64, usize)> {
            let name = format!("blk.0.ffn_{which}_exps.weight");
            let t = src
                .tensor(&name)
                .ok_or_else(|| eyre!("expert pager: tensor `{name}` not found"))?;
            let (k, rows) = role_kr(which);
            let bpe = weight_contract::bytes_per_expert(t.dtype, k, rows)?;
            if t.byte_size as usize != N_EXPERT as usize * bpe {
                return Err(eyre!(
                    "{name}: byte_size {} != {} × {bpe}",
                    t.byte_size,
                    N_EXPERT
                ));
            }
            Ok((t.dtype, k, rows, bpe))
        };
        let g = geom("gate")?;
        let u = geom("up")?;
        let d = geom("down")?;
        let (gate_bpe, up_bpe, down_bpe) = (g.3, u.3, d.3);
        let per_slot = gate_bpe + up_bpe + down_bpe;

        let n_slots = if n_slots > 0 {
            n_slots
        } else if let Some(s) = std::env::var("V41_PAGER_SLOTS").ok().and_then(|v| v.parse().ok()) {
            s
        } else {
            let gb: f64 = std::env::var("V41_PAGER_POOL_GB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60.0);
            let budget = (gb * (1u64 << 30) as f64) as usize;
            // Never exceed the model's total expert count, and always keep enough
            // slots for one token's routed union.
            let max_experts = N_EXPERT as usize * crate::config::N_LAYER as usize;
            (budget / per_slot).clamp(64, max_experts) as u32
        };
        eprintln!(
            "expert pager: {n_slots} slots × {:.1} MB/expert = {:.1} GB pool ({:.1}% of {} total experts)",
            per_slot as f64 / 1e6,
            n_slots as f64 * per_slot as f64 / 1e9,
            100.0 * n_slots as f64 / (N_EXPERT as f64 * crate::config::N_LAYER as f64),
            N_EXPERT as usize * crate::config::N_LAYER as usize,
        );

        let make = |which: (GgufType, u64, u64, usize)| -> eyre::Result<DeviceWeight> {
            let (dtype, k, rows, bpe) = which;
            let buffer = DeviceBuffer::<u8>::new(device_id, n_slots as usize * bpe)?;
            Ok(DeviceWeight {
                buffer,
                n_elements: n_slots as u64 * k * rows,
                dtype,
                shape: vec![n_slots as u64, rows, k],
            })
        };
        let gate = make(g)?;
        let up = make(u)?;
        let down = make(d)?;
        let mut remap_dev = Vec::with_capacity(crate::config::N_LAYER as usize);
        for _ in 0..crate::config::N_LAYER {
            remap_dev.push(DeviceBuffer::<i32>::new(device_id, N_EXPERT as usize)?);
        }
        let routed = RoutedExpertWeights {
            gate,
            up,
            down,
            gate_bytes_per_expert: gate_bpe,
            up_bytes_per_expert: up_bpe,
            down_bytes_per_expert: down_bpe,
            n_slots,
        };
        // === Prefill/decode pool split (M8 step 0) ===
        //
        // Prefill's dense windows live in slots [0, dense_windows*N_EXPERT); decode's
        // LRU allocates strictly above that so it can never clobber a resident prefill
        // window (that would silently compute with another layer's expert weights).
        //
        // The old default was `total_windows - 1`, which handed decode's LRU only
        // `384 + n_slots % 384` slots (384..767, i.e. 7-14 GB) NO MATTER how big
        // `V41_PAGER_POOL_GB` was: raising the pool bought decode literally nothing.
        // The new default is a deliberate budget:
        //
        //  * prefill saturates once every layer it SWEEPS has its own pinned window.
        //    With CED on (`CED_DECODER_START`), the swept set is the encoder layers
        //    0..split plus ONE rotating window for the decoder replay (which runs
        //    once per request, not once per chunk), so windows beyond `split + 1`
        //    buy prefill nothing at all;
        //  * everything above that ceiling is decode's, and on top of it a fixed
        //    fraction of the pool (`V41_PAGER_DECODE_FRAC`, default 0.75) is reserved
        //    for decode even when prefill has not saturated.
        //
        // Why 0.75 (measured 2026-09-13, see docs/v41/DECODE_M8_PLAN.md "Measured"):
        // one window given to prefill saves it (chunks - 1) re-pagings of 7.2 GB;
        // the same window given to decode was worth 13.6 fewer misses/token
        // (~122 ms/token) on a 2702-token prompt. Break-even is ~82 generated
        // tokens, which almost every request clears. Pinning only pays off if the
        // WHOLE encoder fits (21 windows = 151 GB), which no pool on this box
        // affords, so at these sizes prefill's pin line is nearly worthless and
        // decode should take the pool.
        //
        // `V41_PAGER_WINDOWS=<n>` still forces the dense count outright (0 => 1).
        // `V41_PAGER_STRIDE`: slots per pinned window. Default N_EXPERT reproduces
        // the original layout bit-for-bit. Deriving it automatically from box 2's
        // HELLO ownership (`N_EXPERT - owned_count(l)`) needs the remote client,
        // which does not exist yet at pager construction — follow-up.
        let window_stride = std::env::var("V41_PAGER_STRIDE")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(N_EXPERT)
            .clamp(1, N_EXPERT);
        // Windows available at this stride: the last one keeps a full N_EXPERT
        // rotating width, the rest are `window_stride` wide.
        let total_windows = if n_slots > N_EXPERT {
            1 + (n_slots - N_EXPERT) / window_stride
        } else {
            1
        };
        let ced = crate::het::forward_prefill::ced_enabled();
        // +1: the rotating window that serves every layer at or above the pin line.
        let prefill_ceiling = if ced {
            crate::config::CED_DECODER_START as u32 + 2
        } else {
            crate::config::N_LAYER as u32 + 1
        };
        let decode_frac: f64 = std::env::var("V41_PAGER_DECODE_FRAC")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.75)
            .clamp(0.0, 0.95);
        let dense_windows = match std::env::var("V41_PAGER_WINDOWS").ok().and_then(|v| v.parse::<u32>().ok()) {
            Some(0) => 1,
            Some(w) => w.clamp(1, total_windows.max(1)),
            None => {
                let by_frac = ((total_windows as f64) * (1.0 - decode_frac)).floor() as u32;
                by_frac
                    .min(prefill_ceiling)
                    .clamp(1, total_windows.saturating_sub(1).max(1))
            }
        };
        let gb = gate_bpe * pager_read_batch();
        let ub = up_bpe * pager_read_batch();
        let db = down_bpe * pager_read_batch();
        let lru_lo = dense_windows.saturating_sub(1) * window_stride + N_EXPERT;
        // The hot set must fit what this pool keeps resident for decode.
        hot_set::set_capacity(n_slots.saturating_sub(lru_lo) as usize);
        eprintln!(
            "expert pager: {total_windows} windows of {N_EXPERT}; prefill dense = {dense_windows} \
             (pinned layers 0..{}, rest rotate; ced={ced}, ceiling {prefill_ceiling}, decode_frac {decode_frac:.2}), \
             decode LRU = slots {lru_lo}..{n_slots} ({} slots, {:.1} GB)",
            dense_windows.saturating_sub(1),
            n_slots.saturating_sub(lru_lo),
            n_slots.saturating_sub(lru_lo) as f64 * per_slot as f64 / 1e9,
        );
        // GPU repack: three landing zones sized to the largest role, one per
        // role so a miss's three uploads do not serialise on one buffer.
        let (repack, repack_scratch, repack_stream) = if pager_gpu_repack() {
            let max_bpe = gate_bpe.max(up_bpe).max(down_bpe);
            let arch = igpu.properties()?.gcn_arch_name;
            let sc = (0..3)
                .map(|_| DeviceBuffer::<u8>::new(igpu.id, max_bpe))
                .collect::<eyre::Result<Vec<_>>>()?;
            eprintln!(
                "expert pager: GPU MXFP4 repack ON ({arch}, 3 x {:.1} MB scratch)",
                max_bpe as f64 / 1e6
            );
            (Some(Mxfp4Repack::for_arch(&arch)?), sc, Some(Stream::new(igpu.id)?))
        } else {
            (None, Vec::new(), None)
        };
        Ok(Self {
            owner,
            routed,
            n_slots,
            dense_windows,
            window_layer: vec![None; total_windows.max(1) as usize],
            window_dense: vec![false; total_windows.max(1) as usize],
            window_stride,
            par_gate: vec![0u8; gb],
            par_up: vec![0u8; ub],
            par_down: vec![0u8; db],
            res: Residency::new(n_slots as usize),
            miss_hist: (std::env::var("V41_MISS_HIST").as_deref() == Ok("1")).then(|| {
                (0..(crate::config::N_LAYER as usize) * (N_EXPERT as usize))
                    .map(|_| std::sync::atomic::AtomicU32::new(0))
                    .collect()
            }),
            stages: {
                // Per-role sizing is enough for the three separate reads, but a
                // COALESCED read puts ALL THREE roles' nibbles in stage[0] and all
                // three scales in stage[1], so those two need room for the whole
                // expert plus O_DIRECT's alignment padding. stage[2] stays per-role
                // (the verifier's scratch). Costs ~24 MB of pinned memory when the
                // flag is on, nothing when it is off.
                let (w0, w1) = if pager_coalesce() {
                    let whole = gate_bpe + up_bpe + down_bpe + 3 * 4096;
                    (whole, whole)
                } else {
                    (gate_bpe, up_bpe)
                };
                // One set per concurrent miss read (`V41_PAGER_MISS_PAR`).
                let mut v = Vec::new();
                for _ in 0..pager_miss_par() {
                    v.push([
                        v4flash_hip::PinnedBuffer::new_with_flags(
                            w0, v4flash_hip::HIP_HOST_MALLOC_NON_COHERENT)?,
                        v4flash_hip::PinnedBuffer::new_with_flags(
                            w1, v4flash_hip::HIP_HOST_MALLOC_NON_COHERENT)?,
                        v4flash_hip::PinnedBuffer::new_with_flags(
                            down_bpe, v4flash_hip::HIP_HOST_MALLOC_NON_COHERENT)?,
                    ]);
                }
                v
            },
            remap: (0..N_EXPERT as i32).map(|e| -e - 1).collect(),
            remap_dev,
            remap_dirty: None,
            remap_stage: {
                let mut v = Vec::with_capacity(8);
                for _ in 0..8 {
                    v.push(PinnedBuffer::<i32>::new(N_EXPERT as usize)?);
                }
                v
            },
            remap_stage_done: (0..8).map(|_| None).collect(),
            remap_stage_next: 0,
            cur_layer: 0,
            count_as_prefill: false,
            scan_window: None,
            device: igpu,
            prefill_requests: 0,
            prefill_misses: 0,
            decode_requests: 0,
            decode_misses: 0,
            decode_read_ns: 0,
            decode_h2d_ns: 0,
            decode_alloc_ns: 0,
            decode_pread_ns: 0,
            decode_repack_ns: 0,
            decode_pread_bytes: 0,
            decode_weight_ns: 0,
            decode_weight_bytes: 0,
            decode_scale_ns: 0,
            decode_scale_bytes: 0,
            decode_n_layout: 0,
            decode_n_runs: 0,
            decode_n_direct: 0,
            decode_n_raw: 0,
            prefill_read_ns: 0,
            prefill_h2d_ns: 0,
            prefill_alloc_ns: 0,
            prefill_pread_ns: 0,
            prefill_repack_ns: 0,
            prefill_pread_bytes: 0,
            prefill_repack_gpu_ns: 0,
            union_calls: 0,
            union_sum: 0,
            union_max: 0,
            union_min: u32::MAX,
            union_hist: [0; 6],
            repack,
            repack_scratch,
            repack_stream,
            decode_repack_gpu_ns: 0,
            prefetch: None,
            readahead: None,
        })
    }

    /// Upload `remap` into the CURRENT layer's device buffer. Every `ensure*`
    /// sets `cur_layer` first, so this always targets the layer the host just
    /// described — never a layer whose MoE may still be queued.
    /// Mark the current layer's remap as needing an upload. The copy itself is
    /// deferred to `sync_remap_async` (see the `remap_dirty` note).
    fn upload_remap(&mut self) -> eyre::Result<()> {
        self.remap_dirty = Some(self.cur_layer.clamp(0, crate::config::N_LAYER - 1));
        Ok(())
    }

    /// Queue the pending remap upload on `stream`, which MUST be the stream the
    /// MoE dispatch that reads `remap_dev(layer)` runs on. Idempotent and cheap
    /// when nothing is dirty. Callers: every site that is about to hand
    /// `remap_dev(layer)` to a launch.
    pub fn sync_remap_async(&mut self, layer: i32, stream: &Stream) -> eyre::Result<()> {
        let Some(dirty) = self.remap_dirty else { return Ok(()) };
        self.device.set_current()?;
        if dirty != layer {
            // `self.remap` describes `dirty`, not `layer`. Cannot happen on the
            // paths we know (ensure sets cur_layer first), so take the safe road
            // rather than uploading the wrong layer's table: blocking copy to
            // where it belongs, and leave `layer` alone.
            let d = dirty.clamp(0, crate::config::N_LAYER - 1) as usize;
            self.remap_dev[d].copy_from_host(&self.remap)?;
            self.remap_dirty = None;
            return Ok(());
        }
        let l = layer.clamp(0, crate::config::N_LAYER - 1) as usize;
        let i = self.remap_stage_next % self.remap_stage.len().max(1);
        self.remap_stage_next = self.remap_stage_next.wrapping_add(1);
        if let Some(st) = self.remap_stage.get_mut(i) {
            // The slot's previous copy must have landed before it is rewritten.
            if let Some(ev) = self.remap_stage_done.get(i).and_then(|e| e.as_ref()) {
                ev.synchronize()?;
            }
            st.as_mut_slice().copy_from_slice(&self.remap);
            self.remap_dev[l].copy_from_host_async(st.as_slice(), stream)?;
            let ev = match self.remap_stage_done.get_mut(i).and_then(|e| e.take()) {
                Some(ev) => ev,
                None => v4flash_hip::Event::new_no_timing()?,
            };
            ev.record(stream)?;
            if let Some(slot) = self.remap_stage_done.get_mut(i) {
                *slot = Some(ev);
            }
        } else {
            self.remap_dev[l].copy_from_host(&self.remap)?;
        }
        self.remap_dirty = None;
        Ok(())
    }

    /// Blocking flush, for callers with no stream at hand (teardown, tests).
    pub fn sync_remap_blocking(&mut self) -> eyre::Result<()> {
        let Some(dirty) = self.remap_dirty else { return Ok(()) };
        self.device.set_current()?;
        let l = dirty.clamp(0, crate::config::N_LAYER - 1) as usize;
        self.remap_dev[l].copy_from_host(&self.remap)?;
        self.remap_dirty = None;
        Ok(())
    }

    fn remap_dev_cur(&self) -> &DeviceBuffer<i32> {
        &self.remap_dev[self.cur_layer.clamp(0, crate::config::N_LAYER - 1) as usize]
    }

    /// This layer's remap, for the MoE dispatch. Pointer-stable per layer.
    pub fn remap_dev(&self, layer: i32) -> &DeviceBuffer<i32> {
        &self.remap_dev[layer.clamp(0, crate::config::N_LAYER - 1) as usize]
    }

    /// Count the next `ensure`'s requests/misses as PREFILL, not decode.
    pub fn set_count_as_prefill(&mut self, v: bool) {
        self.count_as_prefill = v;
    }

    /// Confine `ensure`'s ALLOCATION to `[lo, hi)` until cleared. Lookup is
    /// unaffected: a hit anywhere in the pool is still a hit.
    pub fn set_scan_window(&mut self, w: Option<(usize, usize)>) {
        self.scan_window = w;
    }

    /// Slot range `ensure` may allocate from: the scan window when set, else
    /// everything above the dense prefill windows.
    fn alloc_bounds(&self) -> (usize, usize) {
        if let Some((lo, hi)) = self.scan_window {
            let hi = hi.min(self.n_slots as usize);
            if lo < hi {
                return (lo, hi);
            }
        }
        let lo = self.dense_slots();
        let lo = if lo >= self.n_slots as usize { 0 } else { lo };
        (lo, self.n_slots as usize)
    }

    /// Mutable twin, for diagnostics that overwrite a layer's remap directly
    /// (`V41_PAGER_RESIDENT`).
    pub fn remap_dev_mut(&mut self, layer: i32) -> &mut DeviceBuffer<i32> {
        &mut self.remap_dev[layer.clamp(0, crate::config::N_LAYER - 1) as usize]
    }

    /// Page in every expert in `ids` for `layer` and return a `global id -> slot`
    /// remap (`-1` for ids not requested). The returned slice is valid until the
    /// next `ensure`. `ids` must have `len() <= n_slots`.
    /// Number of expert slots in the pool.
    pub fn slots(&self) -> u32 {
        self.n_slots
    }

    /// Page ALL `N_EXPERT` experts of `layer` into slot == expert id (a dense
    /// per-layer window), making the pool a drop-in for a resident `routed`
    /// buffer that is indexed by RAW expert id.
    ///
    /// Batched prefill needs this rather than [`Self::ensure`]. Prefill inverts
    /// `d_selected` through a group builder whose `group_count` / `expert_members`
    /// arrays are sized to `N_EXPERT`, so a group id must be a raw expert id in
    /// `0..N_EXPERT`. The LRU in `ensure` hands out slots anywhere in the pool, and
    /// a slot >= N_EXPERT overruns those arrays — which is exactly the garbage this
    /// replaced. With slot == expert id no remap is needed at all and every
    /// downstream dispatch is byte-identical to the resident path.
    ///
    /// A chunk's routed union at B >> 1 is essentially every expert anyway (routing
    /// is flat), so paging the full set costs little over the union and avoids a
    /// host readback of `d_selected` on the critical path.
    pub fn ensure_layer_dense(&mut self, layer: i32) -> eyre::Result<()> {
        self.cur_layer = layer;
        if self.n_slots < N_EXPERT {
            return Err(eyre!(
                "expert pager: dense layer window needs >= {N_EXPERT} slots, pool has {}",
                self.n_slots
            ));
        }
        let w = self.window_of(layer);
        // A dense fill writes all N_EXPERT slots at `base + e`, which OVERFLOWS a
        // packed window into its neighbours. Refuse rather than corrupt: the union
        // path handles every case we actually take (box 1's share is ~116 of 384,
        // far under the 90% dense shortcut).
        if self.window_width(w) < N_EXPERT as usize {
            return Err(eyre!(
                "expert pager: L{layer} dense fill needs {N_EXPERT} slots but window {w}                  is packed to {} (V41_PAGER_STRIDE)",
                self.window_width(w)
            ));
        }
        let base = self.window_base(w);
        self.prefill_requests += N_EXPERT as u64;
        if self.window_layer[w as usize] == Some(layer) && self.window_dense[w as usize] {
            return Ok(()); // whole layer already resident in its own window
        }
        self.device.set_current()?;
        // Clear the window's identity BEFORE touching it: a failure part-way through
        // must not leave it claiming to hold `layer`, or the next chunk would compute
        // with a half-written mix of two layers' experts and never error.
        self.window_layer[w as usize] = None;
        for sl in base..base + N_EXPERT as usize {
            if let Some(old) = self.res.slot_key[sl].take() {
                self.res.slot_of.remove(&old);
            }
        }
        let names = [
            format!("blk.{layer}.ffn_gate_exps.weight"),
            format!("blk.{layer}.ffn_up_exps.weight"),
            format!("blk.{layer}.ffn_down_exps.weight"),
        ];
        let (gbpe, ubpe, dbpe) = (
            self.routed.gate_bytes_per_expert,
            self.routed.up_bytes_per_expert,
            self.routed.down_bytes_per_expert,
        );
        let batch = pager_read_batch();
        let threads = pager_read_threads();
        let (mut read_ns, mut h2d_ns) = (0u64, 0u64);
        let mut e0 = 0usize;
        while e0 < N_EXPERT as usize {
            let n = batch.min(N_EXPERT as usize - e0);
            {
                let owner = &self.owner;
                let names = &names;
                let read_one = move |i: usize, gs: &mut [u8], us: &mut [u8], ds: &mut [u8]| -> eyre::Result<()> {
                    let src = WeightSrc::from(owner);
                    let tg = src.tensor(&names[0]).ok_or_else(|| eyre!("{}", names[0]))?;
                    let tu = src.tensor(&names[1]).ok_or_else(|| eyre!("{}", names[1]))?;
                    let td = src.tensor(&names[2]).ok_or_else(|| eyre!("{}", names[2]))?;
                    src.read_expert_into(tg, i, gs)?;
                    src.read_expert_into(tu, i, us)?;
                    src.read_expert_into(td, i, ds)?;
                    Ok(())
                };
                let g = &mut self.par_gate[..n * gbpe];
                let u = &mut self.par_up[..n * ubpe];
                let d = &mut self.par_down[..n * dbpe];
                let _t_read = std::time::Instant::now();
                if threads <= 1 {
                    for (i, ((gs, us), ds)) in g
                        .chunks_mut(gbpe)
                        .zip(u.chunks_mut(ubpe))
                        .zip(d.chunks_mut(dbpe))
                        .enumerate()
                    {
                        read_one(e0 + i, gs, us, ds)?;
                    }
                } else {
                    let per = n.div_ceil(threads);
                    let err: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
                    std::thread::scope(|sc| {
                        for (t, ((gc, uc), dc)) in g
                            .chunks_mut(per * gbpe)
                            .zip(u.chunks_mut(per * ubpe))
                            .zip(d.chunks_mut(per * dbpe))
                            .enumerate()
                        {
                            let err = &err;
                            sc.spawn(move || {
                                for (i, ((gs, us), ds)) in gc
                                    .chunks_mut(gbpe)
                                    .zip(uc.chunks_mut(ubpe))
                                    .zip(dc.chunks_mut(dbpe))
                                    .enumerate()
                                {
                                    if let Err(e) = read_one(e0 + t * per + i, gs, us, ds) {
                                        *err.lock().unwrap() = Some(format!("{e:#}"));
                                        return;
                                    }
                                }
                            });
                        }
                    });
                    let failed = err.lock().unwrap().take();
                    if let Some(e) = failed {
                        return Err(eyre!("expert pager parallel read: {e}"));
                    }
                }
                read_ns += _t_read.elapsed().as_nanos() as u64;
            }
            // One contiguous device copy per role: batch slots are consecutive.
            let t_h2d = std::time::Instant::now();
            let s0 = base + e0;
            self.routed.gate.buffer
                .slice_view_mut(s0 * gbpe, n * gbpe)
                .copy_from_host(&self.par_gate[..n * gbpe])?;
            self.routed.up.buffer
                .slice_view_mut(s0 * ubpe, n * ubpe)
                .copy_from_host(&self.par_up[..n * ubpe])?;
            self.routed.down.buffer
                .slice_view_mut(s0 * dbpe, n * dbpe)
                .copy_from_host(&self.par_down[..n * dbpe])?;
            h2d_ns += t_h2d.elapsed().as_nanos() as u64;
            for i in 0..n {
                let slot = (s0 + i) as u32;
                self.res.slot_of.insert((layer, (e0 + i) as u32), slot);
                self.res.slot_key[slot as usize] = Some((layer, (e0 + i) as u32));
            }
            self.prefill_misses += n as u64;
            e0 += n;
        }
        self.prefill_read_ns += read_ns;
        self.prefill_h2d_ns += h2d_ns;
        self.window_layer[w as usize] = Some(layer);
        self.window_dense[w as usize] = true;
        Ok(())
    }

    /// Batched miss service: classify ALL of `ids` first, then read every miss
    /// in ONE parallel pass, then upload.
    ///
    /// [`Self::ensure`] walks `ids` serially and services each miss end-to-end
    /// before looking at the next, so a layer's ~6 misses are ~6 sequential
    /// 18.8 MB reads. Measured 11.6-13 ms each at ~1.39 GB/s against an NVMe
    /// that does 4.3 GB/s at depth — the queue is starved because only one
    /// expert (3 role reads) is ever in flight. Classifying first lets all of a
    /// layer's misses queue together, which is the only way to reach the drive's
    /// parallel rate.
    ///
    /// Reuses the `par_*` staging buffers the prefill path already allocates
    /// (`pager_read_batch()` experts' worth, default 32 >> the <=6 a decode
    /// layer can miss).
    ///
    /// `V41_PAGER_BATCH_MISS=0` restores the serial path.
    pub fn ensure_batched(&mut self, layer: i32, ids: &[u32]) -> eyre::Result<&[i32]> {
        let mut c = Claims::default();
        match self.ensure_batched_inner(layer, ids, &mut c) {
            Ok(()) => Ok(&self.remap),
            Err(e) => {
                self.res.release_unlanded(layer, &c);
                Err(e)
            }
        }
    }

    fn ensure_batched_inner(&mut self, layer: i32, ids: &[u32], c: &mut Claims) -> eyre::Result<()> {
        self.cur_layer = layer;
        if ids.len() > self.n_slots as usize {
            return Err(eyre!(
                "expert pager: {} experts requested but only {} slots",
                ids.len(),
                self.n_slots
            ));
        }
        self.device.set_current()?;
        for (e, r) in self.remap.iter_mut().enumerate() {
            *r = -(e as i32) - 1;
        }

        // --- phase 1: classify. Hits are finished here; misses get a slot. ---
        let lru_lo = {
            let lo = self.dense_slots();
            if lo >= self.n_slots as usize { 0 } else { lo }
        };
        let call_tick = self.res.tick;
        for &id in ids {
            if self.count_as_prefill {
                self.prefill_requests += 1;
            } else {
                self.decode_requests += 1;
            }
            let key = (layer, id);
            if let Some(&slot) = self.res.slot_of.get(&key) {
                self.res.touch(slot);
                // ABSOLUTE pool slot, NOT a window-relative one (that is what
                // `write_window_remap` writes into the same field). The group
                // builder's arrays are sized to `sparse_group_bound() ==
                // n_slots` and its `n_expert` argument is a BUFFER LIMIT, not a
                // guard: a slot at or above it is dropped SILENTLY while the
                // reducer still claims its zeroed partial row.
                assert!(
                    slot < self.n_slots,
                    "expert pager: L{layer} e{id} resident at slot {slot} >= pool size {}; the \
                     sparse remap carries ABSOLUTE slots and the group arrays only cover the pool",
                    self.n_slots
                );
                self.remap[id as usize] = -(slot as i32) - 1;
                continue;
            }
            if self.count_as_prefill {
                self.prefill_misses += 1;
            } else {
                self.decode_misses += 1;
            }
            if let Some(h) = self.miss_hist.as_ref() {
                let l = layer.clamp(0, crate::config::N_LAYER - 1) as usize;
                if (id as usize) < N_EXPERT as usize {
                    h[l * N_EXPERT as usize + id as usize]
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            // Claimed AND touched now, so a later miss in this same call cannot
            // pick it (see `Residency::claim`); rolled back by the caller if
            // its bytes never land.
            let slot = self.res.claim(key, lru_lo, self.n_slots as usize, call_tick)?;
            if (slot as usize) < self.dense_slots() {
                if let Some(w) = self.window_of_slot(slot) {
                    if let Some(e) = self.window_layer.get_mut(w as usize) { *e = None; }
                    if let Some(d) = self.window_dense.get_mut(w as usize) { *d = false; }
                }
            }
            c.pending.push((id, slot));
        }
        let misses = &c.pending;
        if misses.is_empty() {
            self.upload_remap()?;
            return Ok(());
        }
        debug_assert!(
            misses.iter().enumerate().all(|(i, &(_, s))| misses[..i].iter().all(|&(_, t)| t != s)),
            "expert pager: L{layer} two misses claimed one slot: {misses:?}"
        );

        // --- phase 2: one parallel read for every miss in this layer ---
        let names = [
            format!("blk.{layer}.ffn_gate_exps.weight"),
            format!("blk.{layer}.ffn_up_exps.weight"),
            format!("blk.{layer}.ffn_down_exps.weight"),
        ];
        let (gbpe, ubpe, dbpe) = (
            self.routed.gate_bytes_per_expert,
            self.routed.up_bytes_per_expert,
            self.routed.down_bytes_per_expert,
        );
        let n = misses.len();
        if n * gbpe > self.par_gate.len() {
            return Err(eyre!(
                "expert pager: {n} misses exceed the {} -expert staging batch",
                self.par_gate.len() / gbpe.max(1)
            ));
        }
        let t_read = std::time::Instant::now();
        let rp0 = v4flash_core::hf_v41::expert_read_profile();
        let sp0 = v4flash_core::hf_v41::expert_read_split();
        let pa0 = v4flash_core::hf_v41::expert_read_paths();
        let gpu_repack = self.repack.is_some();
        {
            let owner = &self.owner;
            let names = &names;
            let ms = &misses;
            let g = &mut self.par_gate[..n * gbpe];
            let u = &mut self.par_up[..n * ubpe];
            let d = &mut self.par_down[..n * dbpe];
            let err: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
            std::thread::scope(|sc| {
                for (i, ((gs, us), ds)) in g
                    .chunks_mut(gbpe)
                    .zip(u.chunks_mut(ubpe))
                    .zip(d.chunks_mut(dbpe))
                    .enumerate()
                {
                    let err = &err;
                    sc.spawn(move || {
                        let src = WeightSrc::from(owner);
                        let id = ms[i].0 as usize;
                        for (name, dst) in [(&names[0], gs), (&names[1], us), (&names[2], ds)] {
                            match src.tensor(name) {
                                None => {
                                    *err.lock().unwrap() = Some(format!("missing {name}"));
                                    return;
                                }
                                Some(t) => {
                                    // HF layout when the iGPU will do the permutation.
                                    let r = if gpu_repack {
                                        src.read_expert_hf_layout(t, id, dst).map(|_| ())
                                    } else {
                                        src.read_expert_into(t, id, dst)
                                    };
                                    if let Err(e) = r {
                                        *err.lock().unwrap() = Some(format!("{e:#}"));
                                        return;
                                    }
                                }
                            }
                        }
                    });
                }
            });
            let failed = err.lock().unwrap().take();
            if let Some(e) = failed {
                return Err(eyre!("expert pager batched miss read: {e}"));
            }
        }
        // File under the phase this call is serving — `decode_requests` /
        // `decode_misses` above already do (`count_as_prefill`), and these used
        // not to, so prefill's read time landed in the decode timers.
        let rp1 = v4flash_core::hf_v41::expert_read_profile();
        let sp1 = v4flash_core::hf_v41::expert_read_split();
        let pa1 = v4flash_core::hf_v41::expert_read_paths();
        let read_ns = t_read.elapsed().as_nanos() as u64;
        if !self.count_as_prefill {
            // Same split the non-batched `ensure` records, so weight+scale
            // account for ALL decode misses instead of only the per-role path.
            self.decode_weight_ns += sp1.0 - sp0.0;
            self.decode_weight_bytes += sp1.1 - sp0.1;
            self.decode_scale_ns += sp1.2 - sp0.2;
            self.decode_scale_bytes += sp1.3 - sp0.3;
            self.decode_n_layout += pa1.0 - pa0.0;
            self.decode_n_runs += pa1.1 - pa0.1;
            self.decode_n_direct += pa1.2 - pa0.2;
            self.decode_n_raw += pa1.3 - pa0.3;
        }
        if self.count_as_prefill {
            self.prefill_read_ns += read_ns;
            self.prefill_alloc_ns += rp1.1 - rp0.1;
            self.prefill_pread_ns += rp1.2 - rp0.2;
            self.prefill_repack_ns += rp1.3 - rp0.3;
            self.prefill_pread_bytes += rp1.4 - rp0.4;
        } else {
            self.decode_read_ns += read_ns;
            self.decode_alloc_ns += rp1.1 - rp0.1;
            self.decode_pread_ns += rp1.2 - rp0.2;
            self.decode_repack_ns += rp1.3 - rp0.3;
            self.decode_pread_bytes += rp1.4 - rp0.4;
        }

        // --- phase 3: upload + bookkeeping ---
        let t_h2d = std::time::Instant::now();
        let mut gpu_ns = 0u64;
        for (i, &(id, slot)) in misses.iter().enumerate() {
            if gpu_repack {
                let (rp, st) = (self.repack.as_ref().unwrap(), self.repack_stream.as_ref().unwrap());
                gpu_ns += Self::upload_and_repack(
                    rp, st, &mut self.repack_scratch, &mut self.routed, slot,
                    [
                        &self.par_gate[i * gbpe..(i + 1) * gbpe],
                        &self.par_up[i * ubpe..(i + 1) * ubpe],
                        &self.par_down[i * dbpe..(i + 1) * dbpe],
                    ],
                )?;
            } else {
                self.routed.gate.buffer
                    .slice_view_mut(slot as usize * gbpe, gbpe)
                    .copy_from_host(&self.par_gate[i * gbpe..(i + 1) * gbpe])?;
                self.routed.up.buffer
                    .slice_view_mut(slot as usize * ubpe, ubpe)
                    .copy_from_host(&self.par_up[i * ubpe..(i + 1) * ubpe])?;
                self.routed.down.buffer
                    .slice_view_mut(slot as usize * dbpe, dbpe)
                    .copy_from_host(&self.par_down[i * dbpe..(i + 1) * dbpe])?;
            }
            // Registered at claim time; nothing since may have taken the slot.
            debug_assert!(
                self.res.slot_key[slot as usize] == Some((layer, id))
                    && self.res.slot_of.get(&(layer, id)) == Some(&slot),
                "expert pager: L{layer} e{id} lost its claim on slot {slot} before landing"
            );
            c.landed += 1;
            self.res.touch(slot);
            // ABSOLUTE pool slot; see the resident-hit branch above.
            assert!(
                slot < self.n_slots,
                "expert pager: L{layer} e{id} paged into slot {slot} >= pool size {}",
                self.n_slots
            );
            self.remap[id as usize] = -(slot as i32) - 1;
        }
        let h2d_ns = t_h2d.elapsed().as_nanos() as u64;
        if self.count_as_prefill {
            self.prefill_h2d_ns += h2d_ns;
            self.prefill_repack_gpu_ns += gpu_ns;
        } else {
            self.decode_h2d_ns += h2d_ns;
            self.decode_repack_gpu_ns += gpu_ns;
        }
        self.upload_remap()?;
        Ok(())
    }

    /// Mark `is_remote` experts as NOT-OURS in the remap that [`Self::ensure`]
    /// just produced, and re-upload. Decode path.
    ///
    /// Must run AFTER `ensure`, and must only TOUCH the remote entries.
    /// `ensure` maps each requested id to an LRU slot (`-(slot)-1`, where slot
    /// != id) and resets everything else to `-(e)-1`. So a remote-owned expert
    /// that was merely filtered out of the `ids` list still reads as
    /// "ours, at slot e" — pointing at whatever the pool happens to hold in slot
    /// e. Silent garbage, not an error. Overwriting those entries with 0
    /// (non-negative => the other device owns it) is what makes the filter safe.
    ///
    /// Contrast the prefill path: there the window is dense with slot == expert
    /// id, so `set_remote_exclusion` can rebuild the whole remap from scratch.
    /// Here it would destroy the LRU slot assignment.
    pub fn mark_remote_after_ensure(
        &mut self,
        is_remote: impl Fn(u32) -> bool,
    ) -> eyre::Result<&DeviceBuffer<i32>> {
        let mut n = 0usize;
        for e in 0..N_EXPERT {
            if is_remote(e) {
                self.remap[e as usize] = 0;
                n += 1;
            }
        }
        if n > 0 {
            self.device.set_current()?;
            self.upload_remap()?;
        }
        Ok(self.remap_dev_cur())
    }

    /// Build and upload the iGPU's REMOTE-EXCLUSION remap for `layer`.
    ///
    /// Entry convention (same one `launch_hetsplit`/`launch_reduce_partials_hetsplit`
    /// already consume in mode 0): a NEGATIVE entry means "this device owns it,
    /// slot = -entry-1"; a NON-NEGATIVE entry means "skip, another device owns
    /// it". With the dense window slot == expert id, so local experts get
    /// `-e-1` — exactly the identity mapping the pager is constructed with —
    /// and remote-owned experts get `0`.
    ///
    /// Callers MUST pin the cap at `N_EXPERT_USED` when using this remap. The
    /// het-split builder's cap is a per-token rank test; a cap below the full
    /// top-k would drop LOCAL picks beyond that rank instead of only remote
    /// ones, silently under-computing the layer.
    pub fn set_remote_exclusion(
        &mut self,
        layer: i32,
        is_remote: impl Fn(u32) -> bool,
    ) -> eyre::Result<&DeviceBuffer<i32>> {
        self.cur_layer = layer;
        // Entries must carry the expert's ACTUAL slot within the window view, not
        // `-e-1`. That identity only held while windows were 384 wide and slot ==
        // expert id; under packing it would point the dispatch at another expert's
        // weights — silently, since the kernel cannot tell a wrong slot from a
        // right one. Anything not resident in THIS window is "not ours" (0), which
        // is also a strict improvement on the old behaviour: unrouted ids used to
        // be marked "ours at stale slot e".
        let w = self.window_of(layer);
        let base = self.window_base(w);
        let width = self.window_width(w);
        for e in 0..N_EXPERT {
            self.remap[e as usize] = if is_remote(e) {
                0
            } else {
                match self.res.slot_of.get(&(layer, e)) {
                    Some(&sl) if (sl as usize) >= base && (sl as usize) < base + width => {
                        -((sl as usize - base) as i32) - 1
                    }
                    _ => 0,
                }
            };
        }
        // The H2D must land on the iGPU: the MoE reads this remap from the iGPU
        // pool, and a copy issued with the dGPU current puts it on the wrong
        // device, where the kernel reads garbage and writes zeros (see the note
        // on `remap_dev`).
        // CONTENT check, distinct from `verify_routing_exactly_once`: that verifies
        // WHICH DEVICE claims each pick; this verifies the slot a claim points at
        // actually holds that expert. A remap entry aimed at the wrong slot passes
        // exactly-once and silently computes from another expert's weights — the
        // precise failure packing introduces. 384 lookups per (layer, chunk).
        for e in 0..N_EXPERT {
            let r = self.remap[e as usize];
            if r >= 0 {
                continue;
            }
            let sl = base + (-r - 1) as usize;
            if self.res.slot_key.get(sl).copied().flatten() != Some((layer, e)) {
                return Err(eyre!(
                    "expert pager: L{layer} remap[{e}]={r} -> slot {sl} holds {:?}, not (L{layer}, {e}).                      Window {w} base {base} width {width} stride {}.",
                    self.res.slot_key.get(sl).copied().flatten(),
                    self.window_stride
                ));
            }
        }
        self.device.set_current()?;
        self.upload_remap()?;
        Ok(self.remap_dev_cur())
    }

    /// Page only `ids` for `layer`, keeping the dense window layout (slot ==
    /// `base + expert id`) so every downstream dispatch is byte-identical to
    /// [`Self::ensure_layer_dense`]: same view, same raw-expert-id indexing, no
    /// remap, nothing packed.
    ///
    /// `ensure_layer_dense` loads all `N_EXPERT` experts on the argument that a
    /// chunk's routed union at B >> 1 is essentially everything. That holds at
    /// large B and fails badly at small B: a prompt of B tokens can route to at
    /// most `B * N_EXPERT_USED` experts per layer, so a 17-token prompt needs at
    /// most 102 of 384 yet paid for all 384 — ~7.2 GB per layer, ~288 GB across
    /// 40 layers, for a 17-token prefill.
    ///
    /// Unrouted slots are left stale, which is safe because the MoE dispatch is
    /// by-expert: an expert with no members in `d_selected` launches no work and
    /// its slot is never read. `window_dense` records that the window is only
    /// partially filled so a later dense request refills it instead of trusting
    /// `window_layer` alone.
    pub fn ensure_layer_union(&mut self, layer: i32, ids: &[u32]) -> eyre::Result<()> {
        self.cur_layer = layer;
        if self.n_slots < N_EXPERT {
            return Err(eyre!(
                "expert pager: dense layer window needs >= {N_EXPERT} slots, pool has {}",
                self.n_slots
            ));
        }
        let w = self.window_of(layer);
        let base = self.window_base(w);
        self.prefill_requests += ids.len() as u64;
        self.device.set_current()?;
        // Reassigning the window to a different layer invalidates every slot in it.
        if self.window_layer[w as usize] != Some(layer) {
            self.window_layer[w as usize] = None;
            self.window_dense[w as usize] = false;
            for sl in base..base + self.window_width(w) {
                if let Some(old) = self.res.slot_key[sl].take() {
                    self.res.slot_of.remove(&old);
                }
            }
            self.window_layer[w as usize] = Some(layer);
        }
        // Deduped union size for this (layer, chunk) — the packing statistic.
        {
            let mut seen = vec![false; N_EXPERT as usize];
            let mut u = 0u32;
            for &e in ids {
                if (e as usize) < seen.len() && !seen[e as usize] {
                    seen[e as usize] = true;
                    u += 1;
                }
            }
            self.union_calls += 1;
            self.union_sum += u as u64;
            self.union_max = self.union_max.max(u);
            self.union_min = self.union_min.min(u);
            self.union_hist[((u as usize) / 64).min(5)] += 1;
            // Temporary: sizes the packed-window slot stride. The MAX is what
            // matters — a packed window must hold the worst-case union or the
            // layer cannot be served from it.
            if self.union_calls % 120 == 0 {
                eprintln!(
                    "[union] calls={} mean={:.1} min={} max={} of {N_EXPERT}                      (packed window would need {} slots, {:.0}% of dense)",
                    self.union_calls,
                    self.union_sum as f64 / self.union_calls as f64,
                    self.union_min,
                    self.union_max,
                    self.union_max,
                    100.0 * self.union_max as f64 / N_EXPERT as f64,
                );
                let tot = self.union_calls as f64;
                let over = |s: usize| -> f64 {
                    100.0 * self.union_hist[s..].iter().sum::<u64>() as f64 / tot
                };
                eprintln!(
                    "[union] hist 0-63:{} 64-127:{} 128-191:{} 192-255:{} 256-319:{} 320+:{} \
                     | exceed stride 192: {:.1}%  256: {:.1}%  320: {:.1}%",
                    self.union_hist[0], self.union_hist[1], self.union_hist[2],
                    self.union_hist[3], self.union_hist[4], self.union_hist[5],
                    over(3), over(4), over(5),
                );
            }
        }
        // PACKED assignment: slots are handed out densely by arrival order, not by
        // expert id, so a window only has to be as wide as the union it holds. The
        // dispatch reads the slot out of `remap[e]`
        // (`moe_group_builder.hip:116` mode 0), so it never needed slot == id.
        let width = self.window_width(w);
        let mut assign: Vec<(u32, usize)> = Vec::with_capacity(ids.len());
        let mut need: Vec<(u32, usize)> = Vec::new();
        let mut next_free = 0usize;
        for &e in ids {
            if e >= N_EXPERT {
                return Err(eyre!("expert pager: expert id {e} >= {N_EXPERT}"));
            }
            if assign.iter().any(|&(x, _)| x == e) {
                continue; // deduped union
            }
            // Already resident IN THIS WINDOW? (a slot elsewhere is not reusable:
            // the dispatch only sees `[base, base+width)` through `routed_window`.)
            if let Some(&sl) = self.res.slot_of.get(&(layer, e)) {
                let sl = sl as usize;
                if sl >= base && sl < base + width && self.res.slot_key[sl] == Some((layer, e)) {
                    assign.push((e, sl));
                    continue;
                }
            }
            while next_free < width && self.res.slot_key[base + next_free].is_some() {
                next_free += 1;
            }
            if next_free >= width {
                // Sized from box 2's ownership, so this means the derivation is
                // wrong — NOT something to paper over by evicting, which would
                // silently drop an expert this same dispatch still needs.
                return Err(eyre!(
                    "expert pager: L{layer} union needs > {width} slots (window {w},                      stride {}). Raise V41_PAGER_STRIDE or give box 2 more of this layer.",
                    self.window_stride
                ));
            }
            let sl = base + next_free;
            next_free += 1;
            assign.push((e, sl));
            need.push((e, sl));
        }
        // The remap IS the slot table now, so publish it even when nothing missed.
        self.write_window_remap(layer, w, &assign)?;
        if need.is_empty() {
            return Ok(());
        }
        let names = [
            format!("blk.{layer}.ffn_gate_exps.weight"),
            format!("blk.{layer}.ffn_up_exps.weight"),
            format!("blk.{layer}.ffn_down_exps.weight"),
        ];
        let (gbpe, ubpe, dbpe) = (
            self.routed.gate_bytes_per_expert,
            self.routed.up_bytes_per_expert,
            self.routed.down_bytes_per_expert,
        );
        let batch = pager_read_batch();
        let threads = pager_read_threads();
        let (mut read_ns, mut h2d_ns) = (0u64, 0u64);
        let mut k0 = 0usize;
        while k0 < need.len() {
            let n = batch.min(need.len() - k0);
            let chunk: &[(u32, usize)] = &need[k0..k0 + n];
            {
                let owner = &self.owner;
                let names = &names;
                let read_one = move |e: u32, gs: &mut [u8], us: &mut [u8], ds: &mut [u8]| -> eyre::Result<()> {
                    let src = WeightSrc::from(owner);
                    let tg = src.tensor(&names[0]).ok_or_else(|| eyre!("{}", names[0]))?;
                    let tu = src.tensor(&names[1]).ok_or_else(|| eyre!("{}", names[1]))?;
                    let td = src.tensor(&names[2]).ok_or_else(|| eyre!("{}", names[2]))?;
                    src.read_expert_into(tg, e as usize, gs)?;
                    src.read_expert_into(tu, e as usize, us)?;
                    src.read_expert_into(td, e as usize, ds)?;
                    Ok(())
                };
                let g = &mut self.par_gate[..n * gbpe];
                let u = &mut self.par_up[..n * ubpe];
                let d = &mut self.par_down[..n * dbpe];
                let _t_read = std::time::Instant::now();
                if threads <= 1 {
                    for (i, ((gs, us), ds)) in g
                        .chunks_mut(gbpe)
                        .zip(u.chunks_mut(ubpe))
                        .zip(d.chunks_mut(dbpe))
                        .enumerate()
                    {
                        read_one(chunk[i].0, gs, us, ds)?;
                    }
                } else {
                    let per = n.div_ceil(threads);
                    let err: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
                    std::thread::scope(|sc| {
                        for (t, ((gc, uc), dc)) in g
                            .chunks_mut(per * gbpe)
                            .zip(u.chunks_mut(per * ubpe))
                            .zip(d.chunks_mut(per * dbpe))
                            .enumerate()
                        {
                            let err = &err;
                            sc.spawn(move || {
                                for (i, ((gs, us), ds)) in gc
                                    .chunks_mut(gbpe)
                                    .zip(uc.chunks_mut(ubpe))
                                    .zip(dc.chunks_mut(dbpe))
                                    .enumerate()
                                {
                                    if let Err(e) = read_one(chunk[t * per + i].0, gs, us, ds) {
                                        *err.lock().unwrap() = Some(format!("{e:#}"));
                                        return;
                                    }
                                }
                            });
                        }
                    });
                    let failed = err.lock().unwrap().take();
                    if let Some(e) = failed {
                        return Err(eyre!("expert pager parallel read: {e}"));
                    }
                }
                read_ns += _t_read.elapsed().as_nanos() as u64;
            }
            // Destinations are scattered (slot == base + expert id), so one copy
            // per expert rather than the dense path's single contiguous run.
            let t_h2d = std::time::Instant::now();
            for (i, &(e, slot)) in chunk.iter().enumerate() {
                self.routed.gate.buffer
                    .slice_view_mut(slot * gbpe, gbpe)
                    .copy_from_host(&self.par_gate[i * gbpe..(i + 1) * gbpe])?;
                self.routed.up.buffer
                    .slice_view_mut(slot * ubpe, ubpe)
                    .copy_from_host(&self.par_up[i * ubpe..(i + 1) * ubpe])?;
                self.routed.down.buffer
                    .slice_view_mut(slot * dbpe, dbpe)
                    .copy_from_host(&self.par_down[i * dbpe..(i + 1) * dbpe])?;
                self.res.slot_of.insert((layer, e), slot as u32);
                self.res.slot_key[slot] = Some((layer, e));
            }
            h2d_ns += t_h2d.elapsed().as_nanos() as u64;
            self.prefill_misses += n as u64;
            k0 += n;
        }
        self.prefill_read_ns += read_ns;
        self.prefill_h2d_ns += h2d_ns;
        Ok(())
    }

    /// Which dense window a layer occupies.
    ///
    /// Prefill sweeps layers 0..N_LAYER in order every chunk, which is the worst case
    /// for LRU: it evicts precisely what is needed soonest, so a cyclic scan hits 0%.
    /// Instead the first `dense_windows-1` layers get a DEDICATED window and are never
    /// evicted, and every remaining layer shares the last dense window as a rotating
    /// stream buffer. That makes the hit rate deterministic at (dense_windows-1)/N_LAYER
    /// rather than luck-dependent.
    /// Publish `layer`'s window contents as the device remap: `remap[e]` carries
    /// the expert's slot RELATIVE to the window view (`-idx-1`), everything else 0.
    fn write_window_remap(
        &mut self,
        layer: i32,
        w: u32,
        assign: &[(u32, usize)],
    ) -> eyre::Result<()> {
        self.cur_layer = layer;
        let base = self.window_base(w);
        let width = self.window_width(w);
        for r in self.remap.iter_mut() {
            *r = 0;
        }
        for &(e, sl) in assign {
            // This field carries a WINDOW-RELATIVE slot here and an ABSOLUTE
            // pool slot in `ensure` -- two slot spaces in one i32. A slot from
            // outside this window would underflow `sl - base` (usize) into a
            // huge id, or alias another expert's slot inside the window.
            assert!(
                sl >= base && sl < base + width,
                "expert pager: window remap for expert {e} got slot {sl}, outside window {w} \
                 [{base}, {}); this remap is WINDOW-RELATIVE, `ensure`'s is ABSOLUTE",
                base + width
            );
            assert!(
                (e as usize) < self.remap.len(),
                "expert pager: window remap for expert id {e} >= remap length {}",
                self.remap.len()
            );
            self.remap[e as usize] = -((sl - base) as i32) - 1;
        }
        self.device.set_current()?;
        self.upload_remap()?;
        Ok(())
    }

    /// Windows that are PINNED to one layer; the last window rotates and keeps a
    /// full `N_EXPERT` stride so unsplit / replay layers (union up to 384) still fit.
    fn pinned_windows(&self) -> u32 {
        self.dense_windows.saturating_sub(1)
    }

    fn window_base(&self, w: u32) -> usize {
        w.min(self.pinned_windows()) as usize * self.window_stride as usize
    }

    fn window_width(&self, w: u32) -> usize {
        if w < self.pinned_windows() {
            self.window_stride as usize
        } else {
            N_EXPERT as usize
        }
    }

    /// First slot of the decode LRU region = end of the last dense window.
    /// Replaces the old `dense_windows * N_EXPERT`, which assumed every window
    /// was 384 wide.
    /// Group-id space the SPARSE (absolute-slot) residency needs.
    ///
    /// `ensure` hands out ABSOLUTE pool slots and writes them into the shared
    /// remap as `-(slot) - 1`, so `moe_group_builder_hetsplit` mode 0 emits a
    /// pool slot as the group id. The builder's `n_expert` argument is a BUFFER
    /// LIMIT, not a guard -- `group_count` / `expert_members` are indexed
    /// `g * max_per_expert + pos` -- so the group arrays and that bound must
    /// cover the WHOLE POOL, not `N_EXPERT`.
    ///
    /// Sizing them to `N_EXPERT` against a multi-thousand-slot pool was #0b:
    /// every sparse-resident routed expert was dropped SILENTLY while
    /// `q2_k_reduce_partials_hetsplit` still claimed its (zeroed) partial row
    /// (it keys on `remap[e] < 0`, which has no bound). Clean inputs, identical
    /// expert ids, clean shared expert, 0.83 relative error in the layer output
    /// and ~60% argmax agreement against decode.
    pub fn sparse_group_bound(&self) -> u32 {
        self.n_slots
    }

    /// Can the sparse view's group ids survive the work-item packing?
    ///
    /// `moe_work_items_builder` packs `(group_id << 16) | member_start` into an
    /// `i32` and the consumers read it back as
    /// `(unsigned)(packed >> 16) & 0xffff`, so ids round-trip for the full
    /// 16-bit range but not beyond it. A pool this large is not reachable today
    /// (65536 slots is ~1.2 TB of experts); the check is here so a future pool
    /// growth fails the predicate and falls back to the dense window instead of
    /// silently aliasing group ids.
    pub fn sparse_group_ids_in_range(&self) -> bool {
        self.n_slots as usize <= MAX_MOE_GROUP_IDS
    }

    fn dense_slots(&self) -> usize {
        let lo = self.pinned_windows() as usize * self.window_stride as usize
            + N_EXPERT as usize;
        lo.min(self.n_slots as usize)
    }

    /// Which window a slot belongs to, or `None` if it is in the LRU region.
    /// Replaces `slot / N_EXPERT`.
    fn window_of_slot(&self, slot: u32) -> Option<u32> {
        let s = slot as usize;
        if s >= self.dense_slots() {
            return None;
        }
        let stride = self.window_stride as usize;
        let pinned = self.pinned_windows() as usize;
        if stride > 0 && s < pinned * stride {
            Some((s / stride) as u32)
        } else {
            Some(pinned as u32)
        }
    }

    fn window_of(&self, layer: i32) -> u32 {
        let pinned = self.dense_windows.saturating_sub(1);
        if pinned > 0 && (layer as u32) < pinned {
            layer as u32
        } else {
            pinned
        }
    }

    /// A `RoutedExpertWeights` VIEW onto `layer`'s dense window, so the MoE indexes it
    /// by raw expert id exactly as it would a resident buffer. This is what lets
    /// different layers stay resident in different windows while the kernel still sees
    /// slot == expert id.
    pub fn routed_window(&self, layer: i32) -> RoutedExpertWeights {
        let w = self.window_of(layer);
        let base = self.window_base(w);
        // The view must be exactly the window's width: `n_slots` on the returned
        // handle is what the dispatch treats as the group-id bound, and a 384-wide
        // view over a 128-wide window would run off into the next layer's slots.
        let n = self.window_width(w);
        let scale = |v: u64| v / self.n_slots as u64 * n as u64;
        let view = |dw: &DeviceWeight, bpe: usize| DeviceWeight {
            buffer: dw.buffer.slice_view(base * bpe, n * bpe),
            n_elements: scale(dw.n_elements),
            dtype: dw.dtype,
            shape: vec![n as u64, dw.shape[1], dw.shape[2]],
        };
        RoutedExpertWeights {
            gate: view(&self.routed.gate, self.routed.gate_bytes_per_expert),
            up: view(&self.routed.up, self.routed.up_bytes_per_expert),
            down: view(&self.routed.down, self.routed.down_bytes_per_expert),
            gate_bytes_per_expert: self.routed.gate_bytes_per_expert,
            up_bytes_per_expert: self.routed.up_bytes_per_expert,
            down_bytes_per_expert: self.routed.down_bytes_per_expert,
            n_slots: n as u32,
        }
    }
    /// Read ONE missing expert into a staging set (pure I/O + CPU repack; runs
    /// for several misses concurrently under `V41_PAGER_MISS_PAR`). Returns the
    /// coalesced offsets when the two-pread path was taken.
    #[allow(clippy::too_many_arguments)]
    fn read_miss_into(
        owner: &V41HfWeights,
        gpu_repack: bool,
        layer: i32,
        id: u32,
        names: &[String; 3],
        b0: &mut [u8],
        b1: &mut [u8],
        b2: &mut [u8],
    ) -> eyre::Result<Option<[(usize, usize, u32, u32); 3]>> {
            // ONE span for all three roles' nibbles + one for their scales.
            // `None` => this checkpoint's planes are not contiguous; fall through
            // to the per-role reads below, which are always correct.
            let mut coalesced: Option<[(usize, usize, u32, u32); 3]> = None;
            if gpu_repack && pager_coalesce() {
                let src = WeightSrc::from(owner);
                let (t0, t1, t2) = (src.tensor(&names[0]), src.tensor(&names[1]), src.tensor(&names[2]));
                if let (Some(t0), Some(t1), Some(t2)) = (t0, t1, t2) {
                    coalesced = src.read_expert_runs_direct(
                        [&t0, &t1, &t2], id as usize,
                        &mut *b0, &mut *b1,
                    )?;
                }
            }
            if let (true, Some(offs)) = (pager_coalesce_check(), coalesced) {
                // Re-read each role the per-role way into stage[2] and compare the
                // bytes the coalesced read placed at its own residues.
                let src = WeightSrc::from(owner);
                for r in 0..3 {
                    let Some(t) = src.tensor(&names[r]) else { continue };
                    let (po, so, out2, nb2) = offs[r];
                    let (plen, slen) = (out2 as usize * nb2 as usize * 16, out2 as usize * nb2 as usize);
                    let mut refbuf = vec![0u8; plen + slen];
                    if src.read_expert_hf_layout(&t, id as usize, &mut refbuf)? {
                        let got_p = &b0[po..po + plen];
                        let got_s = &b1[so..so + slen];
                        if got_p != &refbuf[..plen] {
                            let i = got_p.iter().zip(&refbuf[..plen]).position(|(a, b)| a != b).unwrap_or(0);
                            eprintln!("PAGER_COALESCE_CHECK L{layer} e{id} role{r}: PACKED differs at byte {i} of {plen}");
                        }
                        if got_s != &refbuf[plen..] {
                            let i = got_s.iter().zip(&refbuf[plen..]).position(|(a, b)| a != b).unwrap_or(0);
                            eprintln!("PAGER_COALESCE_CHECK L{layer} e{id} role{r}: SCALE differs at byte {i} of {slen}");
                        }
                    }
                }
            }
            if coalesced.is_some() {
                // bytes already staged; skip the per-role reads
            } else if miss_read_threads() > 1 {
                // MEASUREMENT (M8-E, `V41_PAGER_MISS_THREADS`): the three roles of one
                // miss are three independent 6.3 MB (pread + scalar repack) jobs. Serially
                // they run at ~3.1 GB/s, well under the NVMe's 4.3 GB/s, and the repack is
                // pure single-core CPU — so the serial read is neither at the device floor
                // nor overlapping the CPU with the device. One thread per role is the
                // cheapest test of where the floor actually is. Default 1 = old behaviour.
                let err: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
                std::thread::scope(|sc| {
                    for (name, dst) in [
                        (&names[0], &mut *b0),
                        (&names[1], &mut *b1),
                        (&names[2], &mut *b2),
                    ] {
                        let err = &err;
                        sc.spawn(move || {
                            let src = WeightSrc::from(owner);
                            match src.tensor(name) {
                                None => *err.lock().unwrap() = Some(format!("missing {name}")),
                                Some(t) => {
                                    let r = if gpu_repack {
                                        src.read_expert_hf_layout(t, id as usize, dst).map(|_| ())
                                    } else {
                                        src.read_expert_into(t, id as usize, dst)
                                    };
                                    if let Err(e) = r {
                                        *err.lock().unwrap() = Some(format!("{e:#}"));
                                    }
                                }
                            }
                        });
                    }
                });
                let failed = err.lock().unwrap().take();
                if let Some(e) = failed {
                    return Err(eyre!("expert pager miss read: {e}"));
                }
            } else {
                let src = WeightSrc::from(owner);
                let tg = src.tensor(&names[0]).ok_or_else(|| eyre!("{}", names[0]))?;
                let tu = src.tensor(&names[1]).ok_or_else(|| eyre!("{}", names[1]))?;
                let td = src.tensor(&names[2]).ok_or_else(|| eyre!("{}", names[2]))?;
                if gpu_repack {
                    src.read_expert_hf_layout(tg, id as usize, b0)?;
                    src.read_expert_hf_layout(tu, id as usize, b1)?;
                    src.read_expert_hf_layout(td, id as usize, b2)?;
                } else {
                    src.read_expert_into(tg, id as usize, b0)?;
                    src.read_expert_into(tu, id as usize, b1)?;
                    src.read_expert_into(td, id as usize, b2)?;
                }
            }
        Ok(coalesced)
    }


    /// Upload one miss's three roles in HF layout and permute them into `slot`
    /// on the iGPU.
    ///
    /// An associated fn over disjoint field borrows, not a `&mut self` method:
    /// the batched caller's source bytes live in `self.par_*`, which it would
    /// otherwise be holding immutably while this needs `&mut self.routed`.
    ///
    /// The stream is synchronised before returning because the next miss reuses
    /// the same three scratch buffers, and `copy_from_host` is a blocking
    /// `hipMemcpy` that does NOT wait on pending kernels — without the sync the
    /// next upload would overwrite bytes a repack is still reading.
    /// Repack a miss STRAIGHT OUT OF PINNED STAGING — no H2D.
    ///
    /// `stage[i]` is `hipHostMalloc` memory, which on this APU is the same
    /// physical RAM the iGPU reads, so the preads have already put the bytes
    /// where the kernel wants them. Box 2's twin is `repack_in_place` in
    /// `remote_experts.rs`; this is the back-port. The old
    /// `upload_and_repack` path (copy into `repack_scratch`, then launch) is
    /// kept below for the non-pinned/debug route.
    /// `offs` = `Some([(packed_off, scale_off, rows, nb); 3])` when the three
    /// roles were read COALESCED: all packed nibbles in `stage[0]`, all scales in
    /// `stage[1]`, each role at its own residue. `None` = the per-role layout,
    /// where role `i` owns `stage[i]` with scales following its nibbles.
    ///
    /// Same dispatch box 2 has used since its coalesced path landed; box 1 read
    /// three separate 6.3 MB ranges where one ~17.7 MB range measures +14% on
    /// this device (3,320 -> 3,788 MB/s at bs 6300k vs 18900k) and issues a third
    /// as many requests, which matters more than the bandwidth on a CPU-bound
    /// crypt layer.
    fn repack_in_place(
        rp: &Mxfp4Repack,
        st: &Stream,
        routed: &mut RoutedExpertWeights,
        slot: u32,
        stage: &[v4flash_hip::PinnedBuffer<u8>; 3],
        offs: Option<[(usize, usize, u32, u32); 3]>,
    ) -> eyre::Result<u64> {
        let geom = [
            (N_FF_EXP as u32, (N_EMBD / 32) as u32),
            (N_FF_EXP as u32, (N_EMBD / 32) as u32),
            (N_EMBD as u32, (N_FF_EXP / 32) as u32),
        ];
        let bpe = [
            routed.gate_bytes_per_expert,
            routed.up_bytes_per_expert,
            routed.down_bytes_per_expert,
        ];
        let t = std::time::Instant::now();
        for i in 0..3 {
            let (rows, nb) = geom[i];
            debug_assert_eq!(rows as usize * nb as usize * 17, bpe[i]);
            let dst = match i {
                0 => &mut routed.gate.buffer,
                1 => &mut routed.up.buffer,
                _ => &mut routed.down.buffer,
            };
            match offs {
                Some(o) => {
                    let (po, so, o_rows, o_nb) = o[i];
                    debug_assert_eq!((o_rows, o_nb), (rows, nb));
                    let pbase = stage[0].device_ptr() as *mut u8;
                    let sbase = stage[1].device_ptr() as *mut u8;
                    rp.launch_from_ptrs(
                        st, dst, slot as usize * bpe[i],
                        pbase.wrapping_add(po) as v4flash_hip::sys::hipDeviceptr_t,
                        sbase.wrapping_add(so) as v4flash_hip::sys::hipDeviceptr_t,
                        rows, nb,
                    )?;
                    continue;
                }
                None => {}
            }
            rp.launch_from_ptr(
                st,
                dst,
                slot as usize * bpe[i],
                stage[i].device_ptr(),
                stage[i].len(),
                rows,
                nb,
            )?;
        }
        st.synchronize()?;
        Ok(t.elapsed().as_nanos() as u64)
    }

    #[allow(dead_code)]
    fn upload_and_repack(
        rp: &Mxfp4Repack,
        st: &Stream,
        scratch: &mut [DeviceBuffer<u8>],
        routed: &mut RoutedExpertWeights,
        slot: u32,
        src: [&[u8]; 3],
    ) -> eyre::Result<u64> {
        // (rows, blocks per row) per role: gate/up are [N_FF_EXP, N_EMBD/32],
        // down is [N_EMBD, N_FF_EXP/32]. Same block count, different shape.
        let geom = [
            (N_FF_EXP as u32, (N_EMBD / 32) as u32),
            (N_FF_EXP as u32, (N_EMBD / 32) as u32),
            (N_EMBD as u32, (N_FF_EXP / 32) as u32),
        ];
        let bpe = [
            routed.gate_bytes_per_expert,
            routed.up_bytes_per_expert,
            routed.down_bytes_per_expert,
        ];
        for i in 0..3 {
            let stage = src[i];
            debug_assert_eq!(stage.len(), bpe[i]);
            scratch[i].slice_view_mut(0, stage.len()).copy_from_host(stage)?;
            let (rows, nb) = geom[i];
            debug_assert_eq!(rows as usize * nb as usize * 17, bpe[i]);
            let dst = match i {
                0 => &mut routed.gate.buffer,
                1 => &mut routed.up.buffer,
                _ => &mut routed.down.buffer,
            };
            rp.launch(st, dst, slot as usize * bpe[i], &scratch[i], rows, nb)?;
        }
        let t = std::time::Instant::now();
        st.synchronize()?;
        Ok(t.elapsed().as_nanos() as u64)
    }

    pub fn ensure(&mut self, layer: i32, ids: &[u32]) -> eyre::Result<&[i32]> {
        // Misses are registered in `slot_of` BEFORE their bytes are read, so any
        // error after that (a read / O_DIRECT error, a reader panic, a HIP repack
        // failure) must release the ones that never landed: the scheduler fails
        // only the affected request and keeps this pager, and an unreleased claim
        // is a permanent, most-recently-used "hit" on the evicted expert's weights
        // (or uninitialised memory) for every later request.
        let mut c = Claims::default();
        match self.ensure_inner(layer, ids, &mut c) {
            Ok(()) => Ok(&self.remap),
            Err(e) => {
                self.res.release_unlanded(layer, &c);
                Err(e)
            }
        }
    }

    fn ensure_inner(&mut self, layer: i32, ids: &[u32], c: &mut Claims) -> eyre::Result<()> {
        self.cur_layer = layer;
        if ids.len() > self.n_slots as usize {
            return Err(eyre!(
                "expert pager: {} experts requested but only {} slots",
                ids.len(),
                self.n_slots
            ));
        }
        // Pin the device: the pool + remap_dev copies below are H2D onto the iGPU.
        self.device.set_current()?;
        // Reset to the encoded iGPU self-map (-(e)-1); requested ids get their slot below.
        for (e, r) in self.remap.iter_mut().enumerate() {
            *r = -(e as i32) - 1;
        }
        let names = [
            format!("blk.{layer}.ffn_gate_exps.weight"),
            format!("blk.{layer}.ffn_up_exps.weight"),
            format!("blk.{layer}.ffn_down_exps.weight"),
        ];
        let gpu_repack = self.repack.is_some();
        let call_tick = self.res.tick;
        for &id in ids {
            if self.count_as_prefill {
                self.prefill_requests += 1;
            } else {
                self.decode_requests += 1;
            }
            let key = (layer, id);
            if let Some(&slot) = self.res.slot_of.get(&key) {
                self.res.touch(slot);
                // ABSOLUTE pool slot, NOT a window-relative one (that is what
                // `write_window_remap` writes into the same field). The group
                // builder's arrays are sized to `sparse_group_bound() ==
                // n_slots` and its `n_expert` argument is a BUFFER LIMIT, not a
                // guard: a slot at or above it is dropped SILENTLY while the
                // reducer still claims its zeroed partial row.
                assert!(
                    slot < self.n_slots,
                    "expert pager: L{layer} e{id} resident at slot {slot} >= pool size {}; the \
                     sparse remap carries ABSOLUTE slots and the group arrays only cover the pool",
                    self.n_slots
                );
                self.remap[id as usize] = -(slot as i32) - 1;
                continue;
            }
            if self.count_as_prefill {
                self.prefill_misses += 1;
            } else {
                self.decode_misses += 1;
            }
            if let Some(h) = self.miss_hist.as_ref() {
                let l = layer.clamp(0, crate::config::N_LAYER - 1) as usize;
                if (id as usize) < N_EXPERT as usize {
                    h[l * N_EXPERT as usize + id as usize]
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            // Choose a slot: first free, else the LRU victim — but only ABOVE the dense
            // region. Prefill's windows live in slots [0, dense_windows*N_EXPERT) and it
            // trusts `window_layer` to say a whole layer is resident; if decode's LRU
            // took a slot in there, prefill would keep using the window while one of its
            // experts had been overwritten with another layer's weights — wrong output,
            // no error. Degenerate pools (no room above the dense region) fall back to
            // sharing and invalidate the affected window instead.
            // Allocation bounds: the SCAN WINDOW when prefill set one, else
            // everything above the dense prefill windows. Lookup above is
            // unbounded, so a hit anywhere in the pool is still free — only
            // where a MISS may land is restricted, which is what keeps a
            // prefill scan from evicting decode's warm set.
            let (lru_lo, lru_hi) = self.alloc_bounds();
            // Slot claimed NOW (slot_of + slot_key + touch) so the next miss cannot
            // pick it again; the remap entry is written once the bytes have landed,
            // and `ensure` releases the claim if they never do.
            let slot = self.res.claim(key, lru_lo, lru_hi, call_tick)?;
            // Safety net for the degenerate case above: if this slot does fall inside a
            // dense window, that window is no longer a faithful copy of its layer.
            if (slot as usize) < self.dense_slots() {
                if let Some(w) = self.window_of_slot(slot) {
                    if let Some(e) = self.window_layer.get_mut(w as usize) { *e = None; }
                    if let Some(d) = self.window_dense.get_mut(w as usize) { *d = false; }
                }
            }
            c.pending.push((id, slot));
        }
        // Read the misses `stages.len()` at a time, concurrently, then upload /
        // repack in order (see `pager_miss_par`). In CLAIM order: `c.landed`
        // counts a prefix of `c.pending`.
        let k = self.stages.len().max(1);
        let pending = &c.pending;
        for chunk in pending.chunks(k) {
            let t_read = std::time::Instant::now();
            let rp0 = v4flash_core::hf_v41::expert_read_profile();
            let sp0 = v4flash_core::hf_v41::expert_read_split();
            let pa0 = v4flash_core::hf_v41::expert_read_paths();
            type R = Result<Option<[(usize, usize, u32, u32); 3]>, String>;
            let mut results: Vec<R> = Vec::with_capacity(chunk.len());
            {
                let owner = &self.owner;
                let stages = &mut self.stages;
                let names = &names;
                std::thread::scope(|sc| {
                    let mut hs = Vec::with_capacity(chunk.len());
                    for (&(id, _), st) in chunk.iter().zip(stages.iter_mut()) {
                        let [p0, p1, p2] = st;
                        let (b0, b1, b2) = (p0.as_mut_slice(), p1.as_mut_slice(), p2.as_mut_slice());
                        hs.push(sc.spawn(move || -> R {
                            Self::read_miss_into(owner, gpu_repack, layer, id, names, b0, b1, b2)
                                .map_err(|e| format!("{e:#}"))
                        }));
                    }
                    for h in hs {
                        results.push(h.join().unwrap_or_else(|_| Err("miss reader panicked".into())));
                    }
                });
            }
            let read_ns = t_read.elapsed().as_nanos() as u64;
            let rp1 = v4flash_core::hf_v41::expert_read_profile();
            let sp1 = v4flash_core::hf_v41::expert_read_split();
            let pa1 = v4flash_core::hf_v41::expert_read_paths();
            self.decode_n_layout += pa1.0 - pa0.0;
            self.decode_n_runs += pa1.1 - pa0.1;
            self.decode_n_direct += pa1.2 - pa0.2;
            self.decode_n_raw += pa1.3 - pa0.3;
            self.decode_weight_ns += sp1.0 - sp0.0;
            self.decode_weight_bytes += sp1.1 - sp0.1;
            self.decode_scale_ns += sp1.2 - sp0.2;
            self.decode_scale_bytes += sp1.3 - sp0.3;
            if self.count_as_prefill {
                self.prefill_read_ns += read_ns;
                self.prefill_alloc_ns += rp1.1 - rp0.1;
                self.prefill_pread_ns += rp1.2 - rp0.2;
                self.prefill_repack_ns += rp1.3 - rp0.3;
                self.prefill_pread_bytes += rp1.4 - rp0.4;
            } else {
                self.decode_read_ns += read_ns;
                self.decode_alloc_ns += rp1.1 - rp0.1;
                self.decode_pread_ns += rp1.2 - rp0.2;
                self.decode_repack_ns += rp1.3 - rp0.3;
                self.decode_pread_bytes += rp1.4 - rp0.4;
            }
            for (ci, (&(id, slot), res)) in chunk.iter().zip(results).enumerate() {
                let coalesced = res.map_err(|m| eyre!("expert pager miss read L{layer} e{id}: {m}"))?;
                let t_h2d = std::time::Instant::now();
                let gbpe = self.routed.gate_bytes_per_expert;
                let ubpe = self.routed.up_bytes_per_expert;
                let dbpe = self.routed.down_bytes_per_expert;
                if gpu_repack {
                    let (rp, st) = (self.repack.as_ref().unwrap(), self.repack_stream.as_ref().unwrap());
                    let gpu_ns =
                        Self::repack_in_place(rp, st, &mut self.routed, slot, &self.stages[ci], coalesced)?;
                    if self.count_as_prefill {
                        self.prefill_repack_gpu_ns += gpu_ns;
                    } else {
                        self.decode_repack_gpu_ns += gpu_ns;
                    }
                } else {
                    let st = &self.stages[ci];
                    self.routed.gate.buffer.slice_view_mut(slot as usize * gbpe, gbpe).copy_from_host(st[0].as_slice())?;
                    self.routed.up.buffer.slice_view_mut(slot as usize * ubpe, ubpe).copy_from_host(st[1].as_slice())?;
                    self.routed.down.buffer.slice_view_mut(slot as usize * dbpe, dbpe).copy_from_host(st[2].as_slice())?;
                }
                let h2d_ns = t_h2d.elapsed().as_nanos() as u64;
                if self.count_as_prefill {
                    self.prefill_h2d_ns += h2d_ns;
                } else {
                    self.decode_h2d_ns += h2d_ns;
                }
                assert!(
                    slot < self.n_slots,
                    "expert pager: L{layer} e{id} paged into slot {slot} >= pool size {}",
                    self.n_slots
                );
                self.remap[id as usize] = -(slot as i32) - 1;
                c.landed += 1;
            }
        }
        // Split the blocking H2D out of the rest of `ensure`. With zero misses
        // this 1536-byte copy is the ONLY device op in the call, and the call
        // measures ~10.6 ms/layer-lane -- so this timer says whether a tiny
        // synchronous `hipMemcpy` is stalling on the iGPU's queued MoE work.
        {
            let _t = super::forward_prefill::LayerHostTimer::start(
                &super::forward_prefill::LH_REMAP_H2D,
            );
            self.upload_remap()?;
        }
        Ok(())
    }

    /// Is `(layer, e)` resident in the pool right now? Used by the T2-catch-all
    /// assignment: the hub computes only what it already holds and hands
    /// everything else to box 2, so a hub miss never blocks on the hub's
    /// (dm-crypt) disk. Pure lookup — does NOT page and does NOT touch the LRU.
    pub fn is_resident(&self, layer: i32, e: u32) -> bool {
        self.res.slot_of.contains_key(&(layer, e))
    }

    /// Mark `(layer, e)` most-recently-used if resident; returns whether it was.
    /// Box-2 miss substitution (`het::b2_mirror`) accepts a box-1 substitute at
    /// ROUTE time because it is resident, and the pipelined order runs the
    /// other lane's `ensure` before this lane's, so without this the substitute
    /// can be the victim and the stall merely moves to box 1's disk.
    pub fn touch_resident(&mut self, layer: i32, e: u32) -> bool {
        match self.res.slot_of.get(&(layer, e)).copied() {
            Some(slot) => {
                self.res.touch(slot);
                true
            }
            None => false,
        }
    }

    /// Pool slot of `(layer, e)` if resident (diagnostics).
    pub fn resident_slot(&self, layer: i32, e: u32) -> Option<u32> {
        self.res.slot_of.get(&(layer, e)).copied()
    }

    /// Unused slots in the decode LRU region.
    ///
    /// T2-catch-all uses this as a one-time WARM-UP budget: while the hub's pool
    /// still has free slots a miss is worth paying for (it is a first touch, not
    /// an eviction, so it costs nothing to keep), but once full the hub stops
    /// paging entirely and reassigns misses to box 2. That fills the pool with
    /// real working-set members without a background thread, and after fill the
    /// hub never blocks on its own disk again. It is NOT adaptive — membership
    /// freezes at fill; windowed-LFU promotion is the follow-up.
    pub fn lru_free_slots(&self) -> usize {
        let lo = {
            let l = self.dense_slots();
            if l >= self.n_slots as usize { 0 } else { l }
        };
        self.res.slot_key[lo..].iter().filter(|k| k.is_none()).count()
    }

    /// `V41_T2_CATCHALL=1`: box 2 is a catch-all LRU tier, so the hub stops
    /// synchronously paging and reassigns its misses there instead. Requires the
    /// daemon to run with `--paged` (it advertises ownership of everything).
    /// `V41_T2_CATCHALL=2`: catch-all, but the split is a CONSTANT (everything
    /// routed goes to box 2) instead of a function of residency. Removes the
    /// request-history dependence that made long-context output non-reproducible.
    /// See the long note at the mode-2 branch in `forward_layer.rs`.
    pub fn t2_catchall_deterministic() -> bool {
        static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var("V41_T2_CATCHALL").as_deref() == Ok("2")
        });
        *B
    }

    pub fn t2_catchall() -> bool {
        static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var("V41_T2_CATCHALL").map(|v| v != "0").unwrap_or(false)
        });
        *B
    }

    /// The current iGPU-view remap: NEGATIVE = the iGPU owns this expert
    /// (slot `-e-1`), non-negative = another device's. Read-only view for
    /// `verify_routing_exactly_once`.
    pub fn remap(&self) -> &[i32] {
        &self.remap
    }

    /// Cumulative requests across BOTH phases. Kept only for call sites that want
    /// a grand total; the split fields are what any hit-rate claim must cite.
    pub fn requests(&self) -> u64 {
        self.prefill_requests + self.decode_requests
    }

    /// Cumulative misses across both phases. See [`Self::requests`].
    pub fn misses(&self) -> u64 {
        self.prefill_misses + self.decode_misses
    }

    /// A point-in-time snapshot of every paging counter, for per-request /
    /// per-token deltas.
    /// Summarise the miss histogram, clearing it only when `clear` is set.
    /// `None` unless `V41_MISS_HIST=1`.
    ///
    /// The WINDOW is the whole point. Over ONE request `distinct == total` is
    /// nearly tautological — a pair that misses becomes resident and cannot miss
    /// again in that request — so a per-request window proves only that nothing
    /// thrashes inside a turn. Accumulated across MANY requests the same ratio
    /// separates the two cases that matter:
    ///
    ///   distinct ~= total  -> the same pair never misses twice: COMPULSORY.
    ///                         A tail being streamed once each; no pool size helps.
    ///   distinct <<  total -> pairs recur: CAPACITY. total/distinct is roughly
    ///                         how many times the pool re-fetches what it dropped,
    ///                         and a bigger pool should pay.
    ///
    /// Still never cumulative-forever: `expert_stats.json` is cumulative across
    /// runs and that is exactly what makes it useless for sizing.
    pub fn take_miss_shape(&self, clear: bool) -> Option<MissShape> {
        use std::sync::atomic::Ordering::Relaxed;
        let h = self.miss_hist.as_ref()?;
        let ne = N_EXPERT as usize;
        // swap-to-zero: each dump is a window, and the read clears in one pass.
        let vals: Vec<u32> = if clear {
            h.iter().map(|a| a.swap(0, Relaxed)).collect()
        } else {
            h.iter().map(|a| a.load(Relaxed)).collect()
        };
        let total: u64 = vals.iter().map(|&v| v as u64).sum();
        if total == 0 {
            return Some(MissShape { total: 0, ..Default::default() });
        }
        let distinct = vals.iter().filter(|&&v| v > 0).count() as u32;
        let mut by_layer = vec![0u64; crate::config::N_LAYER as usize];
        for (i, &v) in vals.iter().enumerate() {
            by_layer[i / ne] += v as u64;
        }
        let mut top: Vec<u32> = vals.iter().copied().filter(|&v| v > 0).collect();
        top.sort_unstable_by(|a, b| b.cmp(a));
        let top16: u64 = top.iter().take(16).map(|&v| v as u64).sum();
        Some(MissShape {
            total,
            distinct,
            top16_pct: 100.0 * top16 as f64 / total as f64,
            by_layer,
        })
    }

    pub fn counters(&self) -> PagerCounters {
        PagerCounters {
            prefill_requests: self.prefill_requests,
            prefill_misses: self.prefill_misses,
            prefill_read_ns: self.prefill_read_ns,
            prefill_h2d_ns: self.prefill_h2d_ns,
            decode_requests: self.decode_requests,
            decode_misses: self.decode_misses,
            decode_read_ns: self.decode_read_ns,
            decode_h2d_ns: self.decode_h2d_ns,
            decode_alloc_ns: self.decode_alloc_ns,
            decode_pread_ns: self.decode_pread_ns,
            decode_repack_ns: self.decode_repack_ns,
            decode_pread_bytes: self.decode_pread_bytes,
            decode_weight_ns: self.decode_weight_ns,
            decode_weight_bytes: self.decode_weight_bytes,
            decode_scale_ns: self.decode_scale_ns,
            decode_scale_bytes: self.decode_scale_bytes,
            decode_n_layout: self.decode_n_layout,
            decode_n_runs: self.decode_n_runs,
            decode_n_direct: self.decode_n_direct,
            decode_n_raw: self.decode_n_raw,
        }
    }

    /// Slots decode's LRU may allocate from (everything above the dense windows).
    pub fn decode_slots(&self) -> u32 {
        let lo = self.dense_slots() as u32;
        if lo >= self.n_slots { self.n_slots } else { self.n_slots - lo }
    }

    /// Windows reserved for prefill's dense residency.
    pub fn dense_windows(&self) -> u32 {
        self.dense_windows
    }

    /// The HF source this pager owns. The Engram tables are read from the same
    /// safetensors dir and the pager holds the only handle to it, so gathering an
    /// Engram row goes through here.
    pub fn raw(&self) -> &v4flash_core::SafetensorsDir {
        self.owner.raw()
    }

    /// Reset residency (e.g. between independent sequences). Keeps the buffers.
    pub fn clear(&mut self) {
        self.res.clear();
        // The windows too: a window still flagged dense for its layer while
        // `slot_of` is empty would let `ensure_layer_union` pack new experts
        // over it and a later `ensure_layer_dense` trust it unchanged.
        self.window_layer.iter_mut().for_each(|w| *w = None);
        self.window_dense.iter_mut().for_each(|d| *d = false);
    }

    #[allow(dead_code)]
    fn dtype(&self) -> GgufType {
        self.routed.gate.dtype
    }
}

/// Shape of the miss stream, for `V41_MISS_HIST=1`. See `ExpertPager::miss_hist`.
///
/// Reports the three things that distinguish capacity from policy:
///   * `distinct` — how many (layer, expert) pairs missed at all. Close to the
///     number of pairs the workload touches => capacity. Far below it => a small
///     set is thrashing.
///   * `top16_pct` — share of misses from the 16 worst pairs. High => pinning
///     those would pay; low => the misses are spread and pinning cannot help.
///   * `by_layer` — misses per layer. A flat LRU starves layers whose reuse
///     distance is longer, which reads as concentration here.
#[derive(Debug, Default, Clone)]
pub struct MissShape {
    pub total: u64,
    pub distinct: u32,
    pub top16_pct: f64,
    pub by_layer: Vec<u64>,
}

/// Snapshot of [`ExpertPager`]'s counters. Subtract two snapshots for a
/// per-request or per-token delta.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PagerCounters {
    pub prefill_requests: u64,
    pub prefill_misses: u64,
    pub prefill_read_ns: u64,
    pub prefill_h2d_ns: u64,
    pub decode_requests: u64,
    pub decode_misses: u64,
    pub decode_read_ns: u64,
    pub decode_h2d_ns: u64,
    pub decode_alloc_ns: u64,
    pub decode_pread_ns: u64,
    pub decode_repack_ns: u64,
    pub decode_pread_bytes: u64,
    pub decode_weight_ns: u64,
    pub decode_weight_bytes: u64,
    pub decode_scale_ns: u64,
    pub decode_scale_bytes: u64,
    /// Calls per read function, so the byte mismatch localises to a path.
    pub decode_n_layout: u64,
    pub decode_n_runs: u64,
    pub decode_n_direct: u64,
    pub decode_n_raw: u64,
}

impl std::ops::Sub for PagerCounters {
    type Output = PagerCounters;
    fn sub(self, o: PagerCounters) -> PagerCounters {
        PagerCounters {
            prefill_requests: self.prefill_requests - o.prefill_requests,
            prefill_misses: self.prefill_misses - o.prefill_misses,
            prefill_read_ns: self.prefill_read_ns - o.prefill_read_ns,
            prefill_h2d_ns: self.prefill_h2d_ns - o.prefill_h2d_ns,
            decode_requests: self.decode_requests - o.decode_requests,
            decode_misses: self.decode_misses - o.decode_misses,
            decode_read_ns: self.decode_read_ns - o.decode_read_ns,
            decode_h2d_ns: self.decode_h2d_ns - o.decode_h2d_ns,
            decode_alloc_ns: self.decode_alloc_ns - o.decode_alloc_ns,
            decode_pread_ns: self.decode_pread_ns - o.decode_pread_ns,
            decode_repack_ns: self.decode_repack_ns - o.decode_repack_ns,
            decode_pread_bytes: self.decode_pread_bytes - o.decode_pread_bytes,
            decode_weight_ns: self.decode_weight_ns - o.decode_weight_ns,
            decode_weight_bytes: self.decode_weight_bytes - o.decode_weight_bytes,
            decode_scale_ns: self.decode_scale_ns - o.decode_scale_ns,
            decode_scale_bytes: self.decode_scale_bytes - o.decode_scale_bytes,
            decode_n_layout: self.decode_n_layout - o.decode_n_layout,
            decode_n_runs: self.decode_n_runs - o.decode_n_runs,
            decode_n_direct: self.decode_n_direct - o.decode_n_direct,
            decode_n_raw: self.decode_n_raw - o.decode_n_raw,
        }
    }
}

#[cfg(test)]
mod residency_tests {
    use super::*;

    /// Every `slot_of` entry is backed by `slot_key` and vice versa.
    fn consistent(r: &Residency) {
        for (k, &s) in &r.slot_of {
            assert_eq!(r.slot_key[s as usize], Some(*k), "slot_of {k:?} -> {s} not backed by slot_key");
        }
        for (s, k) in r.slot_key.iter().enumerate() {
            if let Some(k) = k {
                assert_eq!(r.slot_of.get(k), Some(&(s as u32)), "slot_key[{s}] = {k:?} not in slot_of");
            }
        }
    }

    /// A full pool: every slot holds (0, slot), stamped in slot order.
    fn full(n: usize) -> Residency {
        let mut r = Residency::new(n);
        for s in 0..n as u32 {
            r.slot_of.insert((0, s), s);
            r.slot_key[s as usize] = Some((0, s));
            r.touch(s);
        }
        r
    }

    /// The `ensure_batched` bug: several no-free-slot misses in ONE call must get
    /// DIFFERENT victims (the two least recent), not the same one.
    #[test]
    fn misses_in_one_call_get_distinct_victims() {
        let mut r = full(4);
        let t0 = r.tick;
        let a = r.claim((1, 10), 0, 4, t0).unwrap();
        let b = r.claim((1, 11), 0, 4, t0).unwrap();
        assert_eq!((a, b), (0, 1), "the two least recent slots, one each");
        assert!(!r.slot_of.contains_key(&(0, 0)) && !r.slot_of.contains_key(&(0, 1)));
        assert_eq!(r.slot_of.get(&(1, 10)), Some(&0));
        assert_eq!(r.slot_of.get(&(1, 11)), Some(&1));
        consistent(&r);
    }

    /// A range smaller than the call's set refuses rather than evicting a slot the
    /// same call already hit or claimed.
    #[test]
    fn claim_never_evicts_this_calls_slots() {
        let mut r = full(2);
        let t0 = r.tick;
        r.touch(0); // a HIT in this call
        assert_eq!(r.claim((1, 5), 0, 2, t0).unwrap(), 1);
        assert!(r.claim((1, 6), 0, 2, t0).is_err(), "both slots belong to this call");
        assert_eq!(r.slot_of.get(&(0, 0)), Some(&0), "the hit survives");
        assert_eq!(r.slot_of.get(&(1, 5)), Some(&1), "the earlier claim survives");
        consistent(&r);
    }

    /// The `ensure` bug: an error after the claims must leave no entry for the
    /// experts whose bytes never landed, and their slots free.
    #[test]
    fn release_unlanded_rolls_back_only_the_unloaded() {
        let mut r = full(4);
        let t0 = r.tick;
        let mut c = Claims::default();
        for id in [20, 21, 22] {
            let s = r.claim((3, id), 0, 4, t0).unwrap();
            c.pending.push((id, s));
        }
        c.landed = 1; // e20 uploaded, then the read for e21 failed
        r.release_unlanded(3, &c);
        assert!(r.slot_of.contains_key(&(3, 20)));
        assert!(!r.slot_of.contains_key(&(3, 21)) && !r.slot_of.contains_key(&(3, 22)));
        for &(_, s) in &c.pending[1..] {
            assert_eq!(r.slot_key[s as usize], None);
            assert_eq!(r.last_use[s as usize], 0);
        }
        consistent(&r);
        // The released slots are free and reused before any resident is evicted.
        let s = r.claim((3, 21), 0, 4, r.tick).unwrap();
        assert!(c.pending[1..].iter().any(|&(_, p)| p == s));
        assert!(r.slot_of.contains_key(&(0, 3)), "the warm resident was not evicted");
        consistent(&r);
    }

    /// `release` must not drop a key that has since moved to another slot.
    #[test]
    fn release_ignores_a_moved_key() {
        let mut r = Residency::new(2);
        let s0 = r.claim((0, 1), 0, 2, 0).unwrap();
        r.release((0, 1), s0 + 1);
        assert_eq!(r.slot_of.get(&(0, 1)), Some(&s0));
        consistent(&r);
    }
}
