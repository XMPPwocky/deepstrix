//! Multi-stream KV arena (docs/v41/MULTISTREAM_DECODE_PLAN.md 3.2 / 3.7).
//!
//! `HetModelState` holds ONE sequence's KV. For S live streams the batched
//! kernels want every row's KV somewhere in ONE allocation per store, addressed
//! by per-row base arrays (the `*_rows` wrappers; `tests/multistream_row_bases.rs`
//! shows them bit-identical to the single-sequence forms). This module owns:
//!
//!   * per layer, one RAW SWA window buffer of `n_slots * ARENA_RAW_ROWS` rows —
//!     stream `s` owns rows `[s*ARENA_RAW_ROWS, (s+1)*ARENA_RAW_ROWS)`, and its
//!     live window is `[raw_off, raw_off + n_raw)` inside that region exactly as
//!     `HetLayerState` keeps it (monotonic append, compaction at the region end);
//!   * per KV-SOURCE layer (`config::KV_SOURCE_LAYERS`), one compressed store:
//!     `comp_kv` (f16 rows), `index_k` (packed E2M1 keys, one per comp row) and
//!     the compressor accumulators (`ratio * width` floats per stream), with a
//!     first-fit region allocator over comp rows;
//!   * per stream, the counters the kernels' tables are derived from.
//!
//! The buffers live inside a `HetModelState` (`KvArena::state`): layer `l`'s
//! `kv_cache` IS the arena's raw buffer for that layer, and the four KV-source
//! layers carry a `HetCompressorState` whose `state_kv`/`state_score` hold every
//! stream's accumulator block and whose `comp_kv`/`index_k` are the shared
//! stores. That is what lets the batched layer driver
//! (`forward_layer_pre_moe_v2`) run S streams through the SAME `&mut
//! HetLayerState` argument it runs one sequence through, with a
//! `RowLayout::Arena` telling it to take bases and counts from the per-row
//! tables instead of the state's scalars. The state's own counters (`n_raw`,
//! `raw_off`, `n_comp`, `n_index_comp`) are meaningless in the arena and stay 0;
//! `with_kv_source` lends the store to the reuse layers exactly as for one
//! sequence.
//!
//! The raw counters are kept ONCE per stream, not per layer: decode advances
//! every layer's window in lockstep (`forward_token_impl` takes the slot from
//! layer 0 for that reason), and a multi-stream step keeps that invariant.
//!
//! Nothing here launches a kernel except the region compaction copy; the
//! tables are plain host vectors (`RowTables`) the step uploads once into a
//! `RowTablesDev`.
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{Device, DeviceBuffer, Stream};

use crate::config::{
    kv_source_of, CED_DECODER_START, COMPRESS_RATIOS, KV_SOURCE_LAYERS, NEG_INF, N_HEAD_DIM, N_LAYER, SWA_WINDOW,
};
use crate::het::state::{CompKvStore, HetCompressorState, HetLayerState, HetModelState};
use crate::index_kv_e2m1::E2M1_KEY_ROW_BYTES;

/// Raw rows per slot beyond the SWA window. The arena only appends DECODE rows
/// (one per step; a DSpark verify block is `MTP_BLOCK` = 5), never a prefill
/// chunk, so it does not need the single-sequence state's `B_MAX` rows of chunk
/// room (`state::KV_CACHE_ROWS` = 128 + 1024): 1152 -> 256 rows saves ~37 MB of
/// dGPU per slot (40 layers x 896 rows x 1 KiB). A full region costs one
/// `compact_raw` (40 layers x two <=128 KiB D2D copies) every
/// `ARENA_RAW_SLACK` tokens of that stream. Must be >= the rows one stream
/// appends per step (`tables` refuses a window at its region end).
pub const ARENA_RAW_SLACK: usize = 128;
/// Raw rows per slot per layer (`raw_region_base`, `needs_compaction`).
pub const ARENA_RAW_ROWS: usize = SWA_WINDOW as usize + ARENA_RAW_SLACK;

/// Most rows ONE stream may run in one step: its next token plus DSpark draft
/// rows (docs/v41/DSPARK_ARENA_PLAN.md 3.1-3.2). Sizes the per-slot
/// accumulator blocks (`CompStore::blocks_per_slot`).
pub const ARENA_ROWS_PER_STREAM: u32 = 8;

/// Accumulator blocks per slot in a store of `ratio`: one per compressor
/// group a step of `ARENA_ROWS_PER_STREAM` rows can touch. Row `j` at `q =
/// pos + j` state-writes block `q / ratio - pos / ratio` (row `q % ratio`), so
/// the rows of one stream never share a block unless they share a group, and a
/// firing row pools a block that holds its whole group: positions written
/// earlier in the same launch, or (block 0) carried from the previous step.
/// Block 0 is the only one that lives across steps; `accept` moves the live
/// partial group there.
fn blocks_per_slot(ratio: u32) -> u32 {
    1 + (ARENA_ROWS_PER_STREAM - 1).div_ceil(ratio)
}

/// Index into `KvArena::stores` / `RowTables::stores` of the store `layer`
/// reads (its KV source's), `None` for the dense layers.
pub fn store_index_of(layer: usize) -> Option<usize> {
    if COMPRESS_RATIOS[layer] == 0 {
        return None;
    }
    // `kv_source_of` is None for a source layer ITSELF (it owns the store).
    let src = kv_source_of(layer).unwrap_or(layer);
    KV_SOURCE_LAYERS.iter().position(|&l| l as usize == src)
}

/// One stream's region in one compressed store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompRegion {
    /// First row of the region (rows of `comp_kv` / `index_k`).
    pub base: u32,
    /// Rows reserved.
    pub cap: u32,
    /// Rows written so far (`HetCompressorState::n_comp`).
    pub n_comp: u32,
    /// Index keys written so far (`n_index_comp`); tracks `n_comp`.
    pub n_index_comp: u32,
}

/// One live stream's KV counters.
#[derive(Clone, Debug)]
pub struct StreamKv {
    /// Next token's position (the KV position of the row this stream will run).
    pub pos: u32,
    /// Live raw window inside the stream's raw region, per `HetLayerState`, for
    /// the ENCODER layers (`< CED_DECODER_START`).
    pub raw_off: u32,
    pub n_raw: u32,
    /// Same for the DECODER layers: under CED a prefill replays only the last
    /// window of the suffix into them, so their window can be shorter than the
    /// encoder's after a snapshot restore + short suffix. Decode advances both
    /// in lockstep from there on.
    pub raw_off_dec: u32,
    pub n_raw_dec: u32,
    /// One region per KV-source store, in `KV_SOURCE_LAYERS` order.
    pub comp: Vec<CompRegion>,
}

impl StreamKv {
    /// The counters after appending position `pos` (`ratios` in store
    /// order): the raw windows slide once full, a store whose boundary fires
    /// at `pos` gains a comp row and a key. `KvArena::advance` and the
    /// per-row tables of a multi-row step both use this, so row `j` of a step
    /// sees exactly what a one-row step would after `j` advances.
    fn step(&mut self, ratios: impl IntoIterator<Item = u32>) {
        if self.n_raw < SWA_WINDOW {
            self.n_raw += 1;
        } else {
            self.raw_off += 1;
        }
        if self.n_raw_dec < SWA_WINDOW {
            self.n_raw_dec += 1;
        } else {
            self.raw_off_dec += 1;
        }
        for (r, ratio) in self.comp.iter_mut().zip(ratios) {
            if (self.pos + 1) % ratio == 0 {
                r.n_comp += 1;
                r.n_index_comp += 1;
            }
        }
        self.pos += 1;
    }
}

/// First-fit row allocator with a sorted, coalesced free list.
#[derive(Clone, Debug)]
pub struct RowFreeList {
    free: Vec<(u32, u32)>,
}

impl RowFreeList {
    pub fn new(rows: u32) -> Self {
        Self { free: if rows > 0 { vec![(0, rows)] } else { Vec::new() } }
    }
    pub fn carve(&mut self, rows: u32) -> Option<u32> {
        let i = self.free.iter().position(|&(_, len)| len >= rows)?;
        let (base, len) = self.free[i];
        if len == rows {
            self.free.remove(i);
        } else {
            self.free[i] = (base + rows, len - rows);
        }
        Some(base)
    }
    /// Carve exactly `[base, base + rows)` when a free run STARTS at `base` and
    /// holds `rows` (a region growing in place into the run right after it).
    pub fn carve_at(&mut self, base: u32, rows: u32) -> bool {
        let Some(i) = self.free.iter().position(|&(b, len)| b == base && len >= rows) else { return false };
        let len = self.free[i].1;
        if len == rows {
            self.free.remove(i);
        } else {
            self.free[i] = (base + rows, len - rows);
        }
        true
    }
    pub fn give_back(&mut self, base: u32, rows: u32) {
        self.free.push((base, rows));
        self.free.sort_unstable();
        let mut out: Vec<(u32, u32)> = Vec::with_capacity(self.free.len());
        for &(b, l) in &self.free {
            if let Some(last) = out.last_mut() {
                if last.0 + last.1 == b {
                    last.1 += l;
                    continue;
                }
            }
            out.push((b, l));
        }
        self.free = out;
    }
    pub fn fits(&self, rows: u32) -> bool {
        self.free.iter().any(|&(_, l)| l >= rows)
    }
    pub fn free_rows(&self) -> u32 {
        self.free.iter().map(|&(_, l)| l).sum()
    }
    pub fn largest_run(&self) -> u32 {
        self.free.iter().map(|&(_, l)| l).max().unwrap_or(0)
    }
}

/// The bounce copies `(src_row, dst_row, rows)` that move `n` rows from `from`
/// to `to` inside ONE buffer, `chunk` rows at a time, in an order that never
/// overwrites a source row before it has been read: front to back when the
/// rows move down, back to front when they move up (the ranges may overlap).
fn move_chunks(from: u32, to: u32, n: u32, chunk: u32) -> Vec<(u32, u32, u32)> {
    let chunk = chunk.max(1);
    let mut out = Vec::with_capacity(n.div_ceil(chunk) as usize);
    let mut r = 0;
    while r < n {
        let len = (n - r).min(chunk);
        out.push((from + r, to + r, len));
        r += len;
    }
    if to > from {
        out.reverse();
    }
    out
}

/// Where `KvArena::grow` puts one store's regions when it has to compact
/// around region `x`: the regions at or below `x` pack DOWN from row 0
/// (ascending, every move downward), the regions above it pack UP against
/// `rows_cap` (descending, every move upward), which leaves every free row in
/// one run right after `x`. `regs` = `(base, cap)` per live region. Returns the
/// moves in a safe execution order as `(index into regs, new base)` (regions
/// already in place included) and the free run `(base, rows)`.
fn plan_compact_around(regs: &[(u32, u32)], x: usize, rows_cap: u32) -> (Vec<(usize, u32)>, (u32, u32)) {
    let xb = regs[x].0;
    let mut low: Vec<usize> = (0..regs.len()).filter(|&i| regs[i].0 <= xb).collect();
    let mut high: Vec<usize> = (0..regs.len()).filter(|&i| regs[i].0 > xb).collect();
    low.sort_by_key(|&i| regs[i].0);
    high.sort_by_key(|&i| std::cmp::Reverse(regs[i].0));
    let mut moves = Vec::with_capacity(regs.len());
    let mut next = 0u32;
    for i in low {
        moves.push((i, next));
        next += regs[i].1;
    }
    let free_base = next;
    let mut top = rows_cap;
    for i in high {
        top -= regs[i].1;
        moves.push((i, top));
    }
    (moves, (free_base, top - free_base))
}

/// The free runs of a store of `rows_cap` rows holding the regions `regs`
/// (`(base, cap)`, any order; overlaps tolerated): the complement, ascending.
fn free_runs(mut regs: Vec<(u32, u32)>, rows_cap: u32) -> Vec<(u32, u32)> {
    regs.sort_unstable();
    let mut free = Vec::new();
    let mut at = 0u32;
    for (b, c) in regs {
        if b > at {
            free.push((at, b - at));
        }
        at = at.max(b + c);
    }
    if rows_cap > at {
        free.push((at, rows_cap - at));
    }
    free
}

/// How `KvArena::grow` found the rows (the most expensive store's way).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GrowHow {
    /// The free run right after the region was big enough.
    InPlace,
    /// The region moved to a free run that holds the grown size.
    Relocated,
    /// The store was compacted around the region first.
    Compacted,
}

/// One compressed store (one KV-source layer) for all streams: allocator +
/// geometry. Its buffers are `KvArena::state.layers[layer].compressor`.
pub struct CompStore {
    pub layer: usize,
    pub ratio: u32,
    pub width: u32,
    pub rows_cap: u32,
    pub free: RowFreeList,
    /// Accumulator blocks (`ratio * width` floats each) per slot; slot `s`
    /// owns blocks `[s * blocks_per_slot, (s + 1) * blocks_per_slot)` and its
    /// carried state is the first (`blocks_per_slot` fn).
    pub blocks_per_slot: u32,
}

impl CompStore {
    /// First accumulator block of `slot` (the one carried across steps).
    fn slot_block(&self, slot: u32) -> u32 {
        slot * self.blocks_per_slot
    }
}

/// Per-row tables for one step, in the order the rows were given. Host
/// vectors; the step uploads them once (`RowTablesDev::upload`). Names match
/// the kernel parameters. Every value is PRE-step: the row's stream has
/// `n_raw` raw rows and `n_comp` comp rows before this row runs; the driver
/// derives the causal counts (`min(n_raw + 1, SWA_WINDOW)`, `n_comp + fires`).
#[derive(Clone, Debug, Default)]
pub struct RowTables {
    pub pos_per: Vec<i32>,
    /// Raw window per row BEFORE the append: rows valid and the window start in
    /// the LAYER buffer (`slot * ARENA_RAW_ROWS + raw_off`), the same for every
    /// layer.
    pub n_raw_per: Vec<i32>,
    pub n_raw_offset_per: Vec<i32>,
    /// Raw append destination per row (`window start + n_raw`), encoder layers.
    pub slot_per: Vec<i32>,
    /// The decoder layers' (`>= CED_DECODER_START`) window and append slot.
    pub n_raw_per_dec: Vec<i32>,
    pub n_raw_offset_per_dec: Vec<i32>,
    pub slot_per_dec: Vec<i32>,
    /// One entry per KV-source store, `KV_SOURCE_LAYERS` order.
    pub stores: Vec<StoreTables>,
}

#[derive(Clone, Debug, Default)]
pub struct StoreTables {
    /// Comp rows written before this step (the row's own boundary, if it
    /// fires this step, is NOT counted).
    pub n_comp_per: Vec<i32>,
    pub comp_base_per: Vec<i32>,
    /// Same rows as `comp_base_per` (index keys parallel the comp rows), as
    /// the u32 the score kernel takes.
    pub keys_base_per: Vec<u32>,
    /// Float-element base of the row's accumulator block; and the block index
    /// for the pool kernel.
    pub state_base_per: Vec<i32>,
    pub state_idx_per: Vec<i32>,
    /// Rows whose compressor boundary FIRES this step (`(pos + 1) % ratio == 0`),
    /// as row indices into the batch, with — per firing row, same order — the
    /// stream's accumulator block (`state_idx_per[row]`), the comp destination
    /// row (`base + n_comp`) and the boundary's position (`pos + 1 - ratio`,
    /// what decode ropes the pooled row at: forward_layer.rs `comp_pos`).
    pub fire_rows: Vec<i32>,
    pub fire_state_idx: Vec<i32>,
    pub fire_dst_row: Vec<i32>,
    pub fire_comp_pos: Vec<i32>,
}

/// Device copies of the per-row base arrays the `*_rows` launches take.
/// Allocated once for `rows_cap` rows; `upload` refills the used prefix.
pub struct StoreTablesDev {
    pub comp_base_per: DeviceBuffer<i32>,
    pub keys_base_per: DeviceBuffer<u32>,
    pub state_base_per: DeviceBuffer<i32>,
    pub fire_state_idx: DeviceBuffer<i32>,
    pub fire_dst_row: DeviceBuffer<i32>,
}

pub struct RowTablesDev {
    pub rows_cap: u32,
    pub slot_per: DeviceBuffer<i32>,
    pub slot_per_dec: DeviceBuffer<i32>,
    pub stores: Vec<StoreTablesDev>,
}

impl RowTablesDev {
    pub fn alloc(dgpu: Device, rows_cap: u32, n_stores: usize) -> eyre::Result<Self> {
        dgpu.set_current()?;
        let n = rows_cap.max(1) as usize;
        let mut stores = Vec::with_capacity(n_stores);
        for _ in 0..n_stores {
            stores.push(StoreTablesDev {
                comp_base_per: DeviceBuffer::<i32>::new(dgpu.id, n)?,
                keys_base_per: DeviceBuffer::<u32>::new(dgpu.id, n)?,
                state_base_per: DeviceBuffer::<i32>::new(dgpu.id, n)?,
                fire_state_idx: DeviceBuffer::<i32>::new(dgpu.id, n)?,
                fire_dst_row: DeviceBuffer::<i32>::new(dgpu.id, n)?,
            });
        }
        Ok(Self { rows_cap, slot_per: DeviceBuffer::<i32>::new(dgpu.id, n)?, slot_per_dec: DeviceBuffer::<i32>::new(dgpu.id, n)?, stores })
    }

    /// Async copies on `stream`, so they FIFO ahead of the step's launches.
    pub fn upload(&mut self, t: &RowTables, stream: &Stream) -> eyre::Result<()> {
        let b = t.pos_per.len();
        if b > self.rows_cap as usize {
            return Err(eyre!("row tables: {b} rows > capacity {}", self.rows_cap));
        }
        if t.stores.len() != self.stores.len() {
            return Err(eyre!("row tables: {} stores, device has {}", t.stores.len(), self.stores.len()));
        }
        fn up<T: Copy>(dst: &mut DeviceBuffer<T>, src: &[T], stream: &Stream) -> eyre::Result<()> {
            if !src.is_empty() {
                let mut v = dst.slice_view_mut(0, src.len());
                v.copy_from_host_async(src, stream)?;
            }
            Ok(())
        }
        up(&mut self.slot_per, &t.slot_per, stream)?;
        up(&mut self.slot_per_dec, &t.slot_per_dec, stream)?;
        for (d, h) in self.stores.iter_mut().zip(&t.stores) {
            up(&mut d.comp_base_per, &h.comp_base_per, stream)?;
            up(&mut d.keys_base_per, &h.keys_base_per, stream)?;
            up(&mut d.state_base_per, &h.state_base_per, stream)?;
            up(&mut d.fire_state_idx, &h.fire_state_idx, stream)?;
            up(&mut d.fire_dst_row, &h.fire_dst_row, stream)?;
        }
        Ok(())
    }
}

pub struct KvArena {
    pub dgpu: Device,
    pub n_slots: u32,
    /// The buffers, as the layer driver takes them (module doc). Layer `l`:
    /// `kv_cache` = `n_slots * ARENA_RAW_ROWS * N_HEAD_DIM` f16; KV-source
    /// layers: `compressor = Some(..)` with `state_kv`/`state_score` =
    /// `[n_slots * blocks_per_slot, ratio * width]` f32, `comp_kv = F16([rows_cap, width])`,
    /// `index_k = Some([rows_cap, E2M1_KEY_ROW_BYTES])`. Counters stay 0.
    pub state: HetModelState,
    pub stores: Vec<CompStore>,
    streams: Vec<Option<StreamKv>>,
}

impl KvArena {
    /// `comp_rows_cap`: rows per compressed store shared by all streams (a
    /// stream at context `c` needs `ceil(c / ratio)` rows in each store).
    /// `alloc` with a CONTEXT budget: every store gets `ctx_rows_budget /
    /// ratio` rows, so `ctx_rows_budget` positions of context can be live
    /// across all streams whatever the per-layer ratio.
    pub fn alloc_ctx(dgpu: Device, n_slots: u32, ctx_rows_budget: u32) -> eyre::Result<Self> {
        Self::alloc_inner(dgpu, n_slots, |ratio| ctx_rows_budget.div_ceil(ratio).max(1))
    }

    pub fn alloc(dgpu: Device, n_slots: u32, comp_rows_cap: u32) -> eyre::Result<Self> {
        Self::alloc_inner(dgpu, n_slots, |_| comp_rows_cap)
    }

    fn alloc_inner(dgpu: Device, n_slots: u32, cap_of: impl Fn(u32) -> u32) -> eyre::Result<Self> {
        if n_slots == 0 {
            return Err(eyre!("kv arena: n_slots must be >= 1"));
        }
        dgpu.set_current()?;
        let raw_rows = (n_slots as usize) * ARENA_RAW_ROWS;
        let mut stores = Vec::with_capacity(KV_SOURCE_LAYERS.len());
        let mut layers = Vec::with_capacity(N_LAYER as usize);
        for l in 0..N_LAYER as usize {
            let kv_cache = DeviceBuffer::<u16>::new(dgpu.id, raw_rows * N_HEAD_DIM as usize)?;
            let compressor = if KV_SOURCE_LAYERS.iter().any(|&s| s as usize == l) {
                let ratio = COMPRESS_RATIOS[l];
                if ratio == 0 || ratio == 4 {
                    // ratio 4 is V4-Flash's FP8/shuffle shape; the arena's f16
                    // store + `ratio * width` accumulators are V4.1's (1, 2).
                    return Err(eyre!("kv arena: KV-source layer {l} has ratio {ratio} (need 1 or 2)"));
                }
                let width = N_HEAD_DIM;
                let comp_rows_cap = cap_of(ratio);
                let bps = blocks_per_slot(ratio);
                let n_state = (n_slots * bps) as usize * (ratio * width) as usize;
                let mut state_kv = DeviceBuffer::<f32>::new(dgpu.id, n_state)?;
                let mut state_score = DeviceBuffer::<f32>::new(dgpu.id, n_state)?;
                state_kv.copy_from_host(&vec![0f32; n_state])?;
                state_score.copy_from_host(&vec![NEG_INF; n_state])?;
                stores.push(CompStore {
                    layer: l,
                    ratio,
                    width,
                    rows_cap: comp_rows_cap,
                    free: RowFreeList::new(comp_rows_cap),
                    blocks_per_slot: bps,
                });
                Some(HetCompressorState {
                    state_kv,
                    state_score,
                    comp_kv: CompKvStore::F16(DeviceBuffer::<u16>::new(
                        dgpu.id,
                        (comp_rows_cap as usize) * width as usize,
                    )?),
                    n_comp: 0,
                    width,
                    head_dim: N_HEAD_DIM,
                    index_k: Some(DeviceBuffer::<u8>::new(
                        dgpu.id,
                        (comp_rows_cap as usize) * E2M1_KEY_ROW_BYTES,
                    )?),
                    n_index_comp: 0,
                })
            } else {
                None
            };
            layers.push(HetLayerState { kv_cache, n_raw: 0, raw_off: 0, compressor, indexer_compressor: None });
        }
        let state = HetModelState { layers, n_kv_max: cap_of(1) };
        Ok(Self { dgpu, n_slots, state, stores, streams: vec![None; n_slots as usize] })
    }

    pub fn stream(&self, slot: u32) -> Option<&StreamKv> {
        self.streams.get(slot as usize).and_then(|s| s.as_ref())
    }
    pub fn stream_mut(&mut self, slot: u32) -> Option<&mut StreamKv> {
        self.streams.get_mut(slot as usize).and_then(|s| s.as_mut())
    }
    pub fn live(&self) -> usize {
        self.streams.iter().filter(|s| s.is_some()).count()
    }

    /// Admit a stream that will run up to `ctx_cap` positions: a free slot plus
    /// `ceil(ctx_cap / ratio)` rows in every store. Fails without touching
    /// anything if any store cannot fit it (first-fit, no compaction).
    pub fn admit(&mut self, ctx_cap: u32, pos0: u32) -> eyre::Result<u32> {
        let slot = self
            .streams
            .iter()
            .position(|s| s.is_none())
            .ok_or_else(|| eyre!("kv arena: all {} slots live", self.n_slots))? as u32;
        let need: Vec<u32> = self.stores.iter().map(|st| ctx_cap.div_ceil(st.ratio).max(1)).collect();
        if let Some((i, _)) = self.stores.iter().enumerate().find(|(i, st)| !st.free.fits(need[*i])) {
            return Err(eyre!(
                "kv arena: store L{} cannot fit {} rows (free {}, largest run {})",
                self.stores[i].layer,
                need[i],
                self.stores[i].free.free_rows(),
                self.stores[i].free.largest_run()
            ));
        }
        let mut comp = Vec::with_capacity(self.stores.len());
        for (i, st) in self.stores.iter_mut().enumerate() {
            let base = st.free.carve(need[i]).expect("checked above");
            comp.push(CompRegion { base, cap: need[i], n_comp: 0, n_index_comp: 0 });
        }
        self.streams[slot as usize] = Some(StreamKv { pos: pos0, raw_off: 0, n_raw: 0, raw_off_dec: 0, n_raw_dec: 0, comp });
        Ok(slot)
    }

    /// Admit a stream whose prompt was prefilled into `src` (a single-sequence
    /// state, compressors at rest on their source layers) and copy its KV in:
    /// every layer's live raw window to the start of the slot's region, each
    /// store's comp rows / index keys to the carved region and its accumulator
    /// block to the slot's. `pos` is the NEXT position (the prompt length).
    /// D2D copies on `stream`; the caller synchronizes before reading.
    pub fn admit_from_state(
        &mut self,
        src: &HetModelState,
        ctx_cap: u32,
        pos: u32,
        stream: &Stream,
    ) -> eyre::Result<u32> {
        let (n_raw, n_raw_dec) = self.source_windows(src, pos)?;
        let slot = self.admit(ctx_cap.max(pos + 1), pos)?;
        // Any failure below must give the slot back: it used to stay allocated
        // with no Stream owning it, and a parked request retried every tick.
        match self.fill_admitted(src, slot, n_raw, n_raw_dec, stream) {
            Ok(()) => Ok(slot),
            Err(e) => {
                let _ = self.release(slot);
                Err(e)
            }
        }
    }

    /// Reserve a slot and `ceil(ctx_cap / ratio)` rows per store for a stream
    /// whose prompt is still being prefilled: `admit` at position 0, filled
    /// later by `fill_reserved`. The scheduler steps only its own streams, so a
    /// reservation is never a step row; compaction and `grow` move it like any
    /// region (zero rows written).
    pub fn reserve(&mut self, ctx_cap: u32) -> eyre::Result<u32> {
        self.admit(ctx_cap.max(1), 0)
    }

    /// Fill reservation `slot` from the prefilled `src` (as `admit_from_state`)
    /// with `pos` = the prompt length. The reservation must hold `pos + 1`
    /// positions (`grow` it first otherwise). On error the slot stays
    /// reserved: it is the caller's to release.
    pub fn fill_reserved(&mut self, slot: u32, src: &HetModelState, pos: u32, stream: &Stream) -> eyre::Result<()> {
        let (n_raw, n_raw_dec) = self.source_windows(src, pos)?;
        let s = self.stream(slot).ok_or_else(|| eyre!("kv arena: slot {slot} not reserved"))?;
        if s.pos != 0 || s.n_raw != 0 || s.n_raw_dec != 0 || s.comp.iter().any(|r| r.n_comp != 0 || r.n_index_comp != 0) {
            return Err(eyre!("kv arena: slot {slot} is not a fresh reservation"));
        }
        let room = self.reserved_positions(slot);
        if room < pos + 1 {
            return Err(eyre!("kv arena: slot {slot} reserves {room} positions, the prompt needs {}", pos + 1));
        }
        self.fill_admitted(src, slot, n_raw, n_raw_dec, stream)?;
        self.stream_mut(slot).expect("checked").pos = pos;
        Ok(())
    }

    /// Positions `slot`'s regions can hold: the stream can run every position
    /// below this (`min` over stores of `cap * ratio`).
    pub fn reserved_positions(&self, slot: u32) -> u32 {
        self.stream(slot)
            .map(|s| s.comp.iter().zip(&self.stores).map(|(r, st)| r.cap * st.ratio).min().unwrap_or(u32::MAX))
            .unwrap_or(0)
    }

    /// Can `slot` run its next position without a region overflowing? The
    /// exact condition `tables` enforces: every store whose compressor boundary
    /// fires at `pos` has a free row left.
    pub fn can_step(&self, slot: u32) -> bool {
        self.can_step_rows(slot, 1)
    }

    /// `can_step` for `rows` consecutive positions in one step (the next token
    /// plus `rows - 1` draft rows): every boundary among them has a free row.
    /// The raw region is not checked: a full one is compacted before the
    /// tables are built (`compact_for_step`).
    pub fn can_step_rows(&self, slot: u32, rows: u32) -> bool {
        let Some(s) = self.stream(slot) else { return false };
        if rows == 0 || rows > ARENA_ROWS_PER_STREAM {
            return false;
        }
        let mut cur = s.clone();
        for _ in 0..rows {
            if cur.comp.iter().zip(&self.stores).any(|(r, st)| (cur.pos + 1) % st.ratio == 0 && r.n_comp >= r.cap) {
                return false;
            }
            cur.step(self.ratios());
        }
        true
    }

    fn ratios(&self) -> impl Iterator<Item = u32> + '_ {
        self.stores.iter().map(|st| st.ratio)
    }

    /// Would `reserve(ctx_cap)` fit, after a compaction if need be, with
    /// `spare` positions of rows still free in every store afterwards (the
    /// room live streams grow into)?
    pub fn fits_with_spare(&self, ctx_cap: u32, spare: u32) -> bool {
        self.stores.iter().all(|st| {
            st.free.free_rows() >= ctx_cap.div_ceil(st.ratio).max(1) + spare.div_ceil(st.ratio)
        })
    }

    /// Could `reserve(ctx_cap)` EVER fit (every store empty)?
    pub fn could_fit(&self, ctx_cap: u32) -> bool {
        self.stores.iter().all(|st| ctx_cap.div_ceil(st.ratio).max(1) <= st.rows_cap)
    }

    /// Rebuild every store's free list as the complement of the live regions.
    /// `grow` / `compact_stores` call it when a move fails midway (the list is
    /// only rebuilt at their end, and a carved relocation target is not yet
    /// owned); the scheduler calls it after aborting everything.
    pub fn rebuild_free_lists(&mut self) {
        for (si, st) in self.stores.iter_mut().enumerate() {
            let regs: Vec<(u32, u32)> = self.streams.iter().flatten().map(|s| (s.comp[si].base, s.comp[si].cap)).collect();
            st.free = RowFreeList { free: free_runs(regs, st.rows_cap) };
        }
    }

    /// Would `reserve(ctx_cap)` fit right now, without compaction?
    pub fn fits_now(&self, ctx_cap: u32) -> bool {
        self.stores.iter().all(|st| st.free.fits(ctx_cap.div_ceil(st.ratio).max(1)))
    }

    /// Grow `slot`'s regions to hold `ctx_cap` positions. Per store: extend in
    /// place into the free run right after the region, else move the region to
    /// a free run that holds the grown size, else compact the store around it
    /// (`plan_compact_around`) and extend. `Ok(None)` = some store has fewer
    /// free rows than the growth needs; nothing was changed. Copies run through
    /// the bounce buffers on `stream`, which is synchronized before return.
    /// Call between steps, like `compact_stores`.
    pub fn grow(
        &mut self,
        slot: u32,
        ctx_cap: u32,
        stream: &Stream,
        bounce_f16: &mut DeviceBuffer<u16>,
        bounce_u8: &mut DeviceBuffer<u8>,
    ) -> eyre::Result<Option<GrowHow>> {
        let r = self.grow_inner(slot, ctx_cap, stream, bounce_f16, bounce_u8);
        if r.is_err() {
            self.rebuild_free_lists();
        }
        r
    }

    fn grow_inner(
        &mut self,
        slot: u32,
        ctx_cap: u32,
        stream: &Stream,
        bounce_f16: &mut DeviceBuffer<u16>,
        bounce_u8: &mut DeviceBuffer<u8>,
    ) -> eyre::Result<Option<GrowHow>> {
        let comp = self.stream(slot).ok_or_else(|| eyre!("kv arena: slot {slot} not live"))?.comp.clone();
        let need: Vec<u32> = self.stores.iter().map(|st| ctx_cap.div_ceil(st.ratio).max(1)).collect();
        if self.stores.iter().zip(&comp).zip(&need).any(|((st, r), &n)| n.saturating_sub(r.cap) > st.free.free_rows()) {
            return Ok(None);
        }
        self.dgpu.set_current()?;
        let mut how = GrowHow::InPlace;
        let mut copied = false;
        for si in 0..self.stores.len() {
            let r = self.streams[slot as usize].as_ref().expect("live").comp[si];
            if need[si] <= r.cap {
                continue;
            }
            let extra = need[si] - r.cap;
            if self.stores[si].free.carve_at(r.base + r.cap, extra) {
                self.streams[slot as usize].as_mut().expect("live").comp[si].cap = need[si];
                continue;
            }
            if let Some(nb) = self.stores[si].free.carve(need[si]) {
                self.move_region_rows(si, r.base, nb, r.n_comp, r.n_index_comp, stream, bounce_f16, bounce_u8)?;
                self.stores[si].free.give_back(r.base, r.cap);
                let reg = &mut self.streams[slot as usize].as_mut().expect("live").comp[si];
                reg.base = nb;
                reg.cap = need[si];
                how = how.max(GrowHow::Relocated);
                copied = true;
                continue;
            }
            self.compact_store_around(si, slot, stream, bounce_f16, bounce_u8)?;
            let r = self.streams[slot as usize].as_ref().expect("live").comp[si];
            if !self.stores[si].free.carve_at(r.base + r.cap, extra) {
                return Err(eyre!("kv arena: L{} grow after compaction found no run after the region", self.stores[si].layer));
            }
            self.streams[slot as usize].as_mut().expect("live").comp[si].cap = need[si];
            how = GrowHow::Compacted;
            copied = true;
        }
        if copied {
            stream.synchronize()?;
        }
        Ok(Some(how))
    }

    /// Compact store `si` so all its free rows sit in one run right after
    /// `slot`'s region (`plan_compact_around`). Accumulator blocks are per
    /// slot and do not move. The caller synchronizes `stream`.
    fn compact_store_around(
        &mut self,
        si: usize,
        slot: u32,
        stream: &Stream,
        bounce_f16: &mut DeviceBuffer<u16>,
        bounce_u8: &mut DeviceBuffer<u8>,
    ) -> eyre::Result<()> {
        let live: Vec<usize> = (0..self.streams.len()).filter(|&sl| self.streams[sl].is_some()).collect();
        let regs: Vec<(u32, u32)> = live.iter().map(|&sl| {
            let r = self.streams[sl].as_ref().expect("live").comp[si];
            (r.base, r.cap)
        }).collect();
        let x = live.iter().position(|&sl| sl == slot as usize).ok_or_else(|| eyre!("kv arena: slot {slot} not live"))?;
        let (moves, free) = plan_compact_around(&regs, x, self.stores[si].rows_cap);
        for (i, nb) in moves {
            let sl = live[i];
            let r = self.streams[sl].as_ref().expect("live").comp[si];
            if r.base != nb {
                self.move_region_rows(si, r.base, nb, r.n_comp, r.n_index_comp, stream, bounce_f16, bounce_u8)?;
                self.streams[sl].as_mut().expect("live").comp[si].base = nb;
            }
        }
        self.stores[si].free = RowFreeList { free: if free.1 > 0 { vec![free] } else { Vec::new() } };
        Ok(())
    }

    /// Move a region's written comp rows and index keys from row `from` to row
    /// `to` of store `si`, through the bounce buffers (`move_chunks` order, so
    /// overlapping moves in either direction are safe). Async on `stream`.
    #[allow(clippy::too_many_arguments)]
    fn move_region_rows(
        &mut self,
        si: usize,
        from: u32,
        to: u32,
        n_comp: u32,
        n_keys: u32,
        stream: &Stream,
        bounce_f16: &mut DeviceBuffer<u16>,
        bounce_u8: &mut DeviceBuffer<u8>,
    ) -> eyre::Result<()> {
        if from == to {
            return Ok(());
        }
        let width = self.stores[si].width as usize;
        let chunk_rows = (bounce_f16.len() / width).min(bounce_u8.len() / E2M1_KEY_ROW_BYTES).max(1) as u32;
        let l = self.stores[si].layer;
        let cs = self.state.layers[l].compressor.as_mut().expect("arena store");
        if n_comp > 0 {
            let buf = cs.comp_kv.f16_mut().expect("f16 store");
            for (src_row, dst_row, n) in move_chunks(from, to, n_comp, chunk_rows) {
                let (src_row, dst_row, n) = (src_row as usize, dst_row as usize, n as usize);
                {
                    let src = buf.slice_view(src_row * width, n * width);
                    let mut b = bounce_f16.slice_view_mut(0, n * width);
                    b.copy_from_buffer_async(&src, stream)?;
                }
                {
                    let b = bounce_f16.slice_view(0, n * width);
                    let mut dst = buf.slice_view_mut(dst_row * width, n * width);
                    dst.copy_from_buffer_async(&b, stream)?;
                }
            }
        }
        if n_keys > 0 {
            let kb = cs.index_k.as_mut().expect("keys");
            for (src_row, dst_row, n) in move_chunks(from, to, n_keys, chunk_rows) {
                let (src_row, dst_row, n) = (src_row as usize, dst_row as usize, n as usize);
                {
                    let src = kb.slice_view(src_row * E2M1_KEY_ROW_BYTES, n * E2M1_KEY_ROW_BYTES);
                    let mut b = bounce_u8.slice_view_mut(0, n * E2M1_KEY_ROW_BYTES);
                    b.copy_from_buffer_async(&src, stream)?;
                }
                {
                    let b = bounce_u8.slice_view(0, n * E2M1_KEY_ROW_BYTES);
                    let mut dst = kb.slice_view_mut(dst_row * E2M1_KEY_ROW_BYTES, n * E2M1_KEY_ROW_BYTES);
                    dst.copy_from_buffer_async(&b, stream)?;
                }
            }
        }
        Ok(())
    }

    /// The live raw windows of a prefilled single-sequence `src` at next
    /// position `pos`: `(n_raw, n_raw_dec)` after checking they are in lockstep
    /// within the encoder / decoder groups and fit the window.
    fn source_windows(&self, src: &HetModelState, pos: u32) -> eyre::Result<(u32, u32)> {
        if src.layers.len() != self.state.layers.len() {
            return Err(eyre!("kv arena: source state has {} layers, arena {}", src.layers.len(), self.state.layers.len()));
        }
        let split = CED_DECODER_START.min(src.layers.len());
        let n_raw = src.layers[0].n_raw;
        let n_raw_dec = src.layers.get(split).map(|l| l.n_raw).unwrap_or(n_raw);
        if src.layers[..split].iter().any(|l| l.n_raw != n_raw) || src.layers[split..].iter().any(|l| l.n_raw != n_raw_dec) {
            return Err(eyre!("kv arena: source state's raw windows are not in lockstep within the encoder/decoder groups"));
        }
        if n_raw > SWA_WINDOW || pos < n_raw || n_raw_dec > SWA_WINDOW || pos < n_raw_dec {
            return Err(eyre!("kv arena: source windows {n_raw}/{n_raw_dec} rows at pos {pos}"));
        }
        Ok((n_raw, n_raw_dec))
    }

    /// Copy `src`'s live KV into freshly admitted `slot` (`admit_from_state`).
    fn fill_admitted(&mut self, src: &HetModelState, slot: u32, n_raw: u32, n_raw_dec: u32, stream: &Stream) -> eyre::Result<()> {
        let hd = N_HEAD_DIM as usize;
        self.dgpu.set_current()?;
        let region = Self::raw_region_base(slot) as usize;
        for (dst, s) in self.state.layers.iter_mut().zip(&src.layers) {
            if s.n_raw == 0 {
                continue;
            }
            let win = s.n_raw as usize * hd;
            let sv = s.kv_cache.slice_view(s.raw_off as usize * hd, win);
            let mut dv = dst.kv_cache.slice_view_mut(region * hd, win);
            dv.copy_from_buffer_async(&sv, stream)?;
        }
        let mut comp = Vec::with_capacity(self.stores.len());
        for (si, st) in self.stores.iter().enumerate() {
            let l = st.layer;
            let scs = src.layers[l].compressor.as_ref().ok_or_else(|| {
                eyre!("kv arena: source state has no compressor at rest on L{l} (lent to a reuse layer?)")
            })?;
            let dcs = self.state.layers[l].compressor.as_mut().expect("arena store");
            let region = self.streams[slot as usize].as_ref().expect("just admitted").comp[si];
            // Keys must cover EVERY comp row when the layer has a key store: the
            // indexer scores `n_comp` rows, so a gap would be scored as whatever
            // the region held before (another stream's keys).
            if scs.n_comp > region.cap || scs.n_index_comp > scs.n_comp
                || (scs.index_k.is_some() && scs.n_index_comp != scs.n_comp)
            {
                return Err(eyre!(
                    "kv arena: L{l} source has {} comp rows ({} keys), region holds {}",
                    scs.n_comp, scs.n_index_comp, region.cap
                ));
            }
            let width = st.width as usize;
            if scs.width != st.width {
                return Err(eyre!("kv arena: L{l} source width {} != arena {}", scs.width, st.width));
            }
            if scs.n_comp > 0 {
                let sbuf = scs.comp_kv.f16().ok_or_else(|| eyre!("kv arena: L{l} source store is not f16"))?;
                let dbuf = dcs.comp_kv.f16_mut().expect("arena store is f16");
                let n = scs.n_comp as usize * width;
                let mut dv = dbuf.slice_view_mut(region.base as usize * width, n);
                dv.copy_from_buffer_async(&sbuf.slice_view(0, n), stream)?;
            }
            if scs.n_index_comp > 0 {
                let sk = scs.index_k.as_ref().ok_or_else(|| eyre!("kv arena: L{l} source has no index keys"))?;
                let dk = dcs.index_k.as_mut().expect("arena store has keys");
                let n = scs.n_index_comp as usize * E2M1_KEY_ROW_BYTES;
                let mut dv = dk.slice_view_mut(region.base as usize * E2M1_KEY_ROW_BYTES, n);
                dv.copy_from_buffer_async(&sk.slice_view(0, n), stream)?;
            }
            let block = (st.ratio * st.width) as usize;
            if scs.state_kv.len() != block {
                return Err(eyre!("kv arena: L{l} source accumulator has {} floats, arena block {block}", scs.state_kv.len()));
            }
            {
                let at = st.slot_block(slot) as usize * block;
                let mut dv = dcs.state_kv.slice_view_mut(at, block);
                dv.copy_from_buffer_async(&scs.state_kv.slice_view(0, block), stream)?;
                let mut dv = dcs.state_score.slice_view_mut(at, block);
                dv.copy_from_buffer_async(&scs.state_score.slice_view(0, block), stream)?;
            }
            comp.push(CompRegion { n_comp: scs.n_comp, n_index_comp: scs.n_index_comp, ..region });
        }
        let s = self.streams[slot as usize].as_mut().expect("just admitted");
        s.n_raw = n_raw;
        s.raw_off = 0;
        s.n_raw_dec = n_raw_dec;
        s.raw_off_dec = 0;
        s.comp = comp;
        Ok(())
    }

    /// Inverse of `admit_from_state`: copy `slot`'s live KV into a single-sequence
    /// state (raw windows compacted to `[0, n_raw)`, comp rows / keys /
    /// accumulators to the store's start) and set its counters, so the existing
    /// single-state consumers (snapshot save, verify) can read it. `dst` must be
    /// at rest (compressors on their source layers) and large enough. D2D on
    /// `stream`; the caller synchronizes.
    pub fn export_to_state(&self, slot: u32, dst: &mut HetModelState, stream: &Stream) -> eyre::Result<()> {
        let s = self.stream(slot).ok_or_else(|| eyre!("kv arena: slot {slot} not live"))?.clone();
        if dst.layers.len() != self.state.layers.len() {
            return Err(eyre!("kv arena: export target has {} layers, arena {}", dst.layers.len(), self.state.layers.len()));
        }
        let hd = N_HEAD_DIM as usize;
        self.dgpu.set_current()?;
        let region = Self::raw_region_base(slot) as usize;
        for (l, (src, d)) in self.state.layers.iter().zip(dst.layers.iter_mut()).enumerate() {
            let (n_raw, raw_off) = if l < CED_DECODER_START { (s.n_raw, s.raw_off) } else { (s.n_raw_dec, s.raw_off_dec) };
            if n_raw > 0 {
                let win = n_raw as usize * hd;
                if d.kv_cache.len() < win {
                    return Err(eyre!("kv arena: export target raw cache too small"));
                }
                let sv = src.kv_cache.slice_view((region + raw_off as usize) * hd, win);
                let mut dv = d.kv_cache.slice_view_mut(0, win);
                dv.copy_from_buffer_async(&sv, stream)?;
            }
            d.n_raw = n_raw;
            d.raw_off = 0;
        }
        for (si, st) in self.stores.iter().enumerate() {
            let l = st.layer;
            let r = s.comp[si];
            let scs = self.state.layers[l].compressor.as_ref().expect("arena store");
            let dcs = dst.layers[l].compressor.as_mut().ok_or_else(|| {
                eyre!("kv arena: export target has no compressor at rest on L{l}")
            })?;
            let width = st.width as usize;
            if r.n_comp > 0 {
                let sbuf = scs.comp_kv.f16().expect("arena store is f16");
                let dbuf = dcs.comp_kv.f16_mut().ok_or_else(|| eyre!("kv arena: L{l} export target store is not f16"))?;
                let n = r.n_comp as usize * width;
                if dbuf.len() < n {
                    return Err(eyre!("kv arena: L{l} export target store too small ({} < {n})", dbuf.len()));
                }
                let mut dv = dbuf.slice_view_mut(0, n);
                dv.copy_from_buffer_async(&sbuf.slice_view(r.base as usize * width, n), stream)?;
            }
            if r.n_index_comp > 0 {
                let sk = scs.index_k.as_ref().expect("arena store has keys");
                let dk = dcs.index_k.as_mut().ok_or_else(|| eyre!("kv arena: L{l} export target has no index keys"))?;
                let n = r.n_index_comp as usize * E2M1_KEY_ROW_BYTES;
                let mut dv = dk.slice_view_mut(0, n);
                dv.copy_from_buffer_async(&sk.slice_view(r.base as usize * E2M1_KEY_ROW_BYTES, n), stream)?;
            }
            let block = (st.ratio * st.width) as usize;
            if dcs.state_kv.len() != block {
                return Err(eyre!("kv arena: L{l} export target accumulator has {} floats, arena block {block}", dcs.state_kv.len()));
            }
            {
                let at = st.slot_block(slot) as usize * block;
                let mut dv = dcs.state_kv.slice_view_mut(0, block);
                dv.copy_from_buffer_async(&scs.state_kv.slice_view(at, block), stream)?;
                let mut dv = dcs.state_score.slice_view_mut(0, block);
                dv.copy_from_buffer_async(&scs.state_score.slice_view(at, block), stream)?;
            }
            dcs.n_comp = r.n_comp;
            dcs.n_index_comp = r.n_index_comp;
        }
        Ok(())
    }

    /// Would `admit(ctx_cap, ..)` fit if the stores were defragmented?
    pub fn fits_after_compaction(&self, ctx_cap: u32) -> bool {
        self.stores.iter().all(|st| st.free.free_rows() >= ctx_cap.div_ceil(st.ratio).max(1))
    }

    /// Slide every live region of every store down to make the free space one
    /// contiguous run at the end. D2D through `bounce` (>= 4096 * width f16 and
    /// >= 4096 * E2M1_KEY_ROW_BYTES bytes; the raw ring scratch is too small,
    /// pass a dedicated buffer). Nothing may be reading the stores (call it
    /// between steps, on the scheduler thread); `stream` is synchronized before
    /// return. Accumulator blocks are per slot and do not move.
    pub fn compact_stores(&mut self, stream: &Stream, bounce_f16: &mut DeviceBuffer<u16>, bounce_u8: &mut DeviceBuffer<u8>) -> eyre::Result<()> {
        let r = self.compact_stores_inner(stream, bounce_f16, bounce_u8);
        if r.is_err() {
            self.rebuild_free_lists();
        }
        r
    }

    fn compact_stores_inner(&mut self, stream: &Stream, bounce_f16: &mut DeviceBuffer<u16>, bounce_u8: &mut DeviceBuffer<u8>) -> eyre::Result<()> {
        self.dgpu.set_current()?;
        let n_stores = self.stores.len();
        for si in 0..n_stores {
            // (slot, base, cap) of every live region, ascending by base.
            let mut regs: Vec<(usize, u32, u32)> = self.streams.iter().enumerate()
                .filter_map(|(sl, s)| s.as_ref().map(|s| (sl, s.comp[si].base, s.comp[si].cap)))
                .collect();
            regs.sort_by_key(|&(_, b, _)| b);
            let mut next_base: u32 = 0;
            for (sl, base, cap) in regs {
                if base != next_base {
                    debug_assert!(base > next_base);
                    let r = self.streams[sl].as_ref().expect("live").comp[si];
                    self.move_region_rows(si, base, next_base, r.n_comp, r.n_index_comp, stream, bounce_f16, bounce_u8)?;
                    self.streams[sl].as_mut().expect("live").comp[si].base = next_base;
                }
                next_base += cap;
            }
            let rows_cap = self.stores[si].rows_cap;
            self.stores[si].free = RowFreeList { free: if rows_cap > next_base { vec![(next_base, rows_cap - next_base)] } else { Vec::new() } };
        }
        stream.synchronize()?;
        Ok(())
    }

    pub fn release(&mut self, slot: u32) -> eyre::Result<()> {
        let s = self.streams.get_mut(slot as usize).and_then(|s| s.take())
            .ok_or_else(|| eyre!("kv arena: slot {slot} not live"))?;
        for (st, r) in self.stores.iter_mut().zip(&s.comp) {
            st.free.give_back(r.base, r.cap);
        }
        Ok(())
    }

    /// Row start of `slot`'s raw region in every layer buffer, in rows.
    pub fn raw_region_base(slot: u32) -> u32 {
        slot * ARENA_RAW_ROWS as u32
    }

    /// The step's tables for `slots`, one row per entry, in order. A stream
    /// may run several CONSECUTIVE rows (its next token, then draft rows at the
    /// following positions; `ARENA_ROWS_PER_STREAM` at most): row `j` of a run
    /// gets the tables a one-row step would after `j` advances (`StreamKv::
    /// step`), so it attends to the rows before it in the same step and to
    /// nothing after it. Its accumulator rows go to the block of its own
    /// compressor group (`blocks_per_slot`). A slot that reappears after
    /// another slot's rows is refused. The counters do not move: the caller
    /// `accept`s each stream's kept rows after the step.
    pub fn tables(&self, slots: &[u32]) -> eyre::Result<RowTables> {
        let mut t = RowTables { stores: vec![StoreTables::default(); self.stores.len()], ..Default::default() };
        let mut cur: Option<(u32, StreamKv, u32)> = None; // (slot, counters at row j, pre-step pos)
        let mut j = 0u32;
        for (b, &slot) in slots.iter().enumerate() {
            match cur.as_mut() {
                Some((s0, c, _)) if *s0 == slot => {
                    c.step(self.stores.iter().map(|st| st.ratio));
                    j += 1;
                    if j >= ARENA_ROWS_PER_STREAM {
                        return Err(eyre!("kv arena: slot {slot} runs more than {ARENA_ROWS_PER_STREAM} rows in one step"));
                    }
                }
                _ => {
                    if slots[..b].contains(&slot) {
                        return Err(eyre!("kv arena: slot {slot} appears twice in the step, not in consecutive rows"));
                    }
                    let s = self.stream(slot).ok_or_else(|| eyre!("kv arena: slot {slot} not live"))?;
                    cur = Some((slot, s.clone(), s.pos));
                    j = 0;
                }
            }
            let (_, s, pos0) = cur.as_ref().expect("set above");
            // The append would land past the slot's raw region, in the NEXT
            // slot's window (the layer buffer's own bound would not catch it).
            if (s.raw_off + s.n_raw) as usize >= ARENA_RAW_ROWS || (s.raw_off_dec + s.n_raw_dec) as usize >= ARENA_RAW_ROWS {
                return Err(eyre!("kv arena: slot {slot} raw window is at its region end (compact_raw first)"));
            }
            let region = Self::raw_region_base(slot);
            t.pos_per.push(s.pos as i32);
            t.n_raw_per.push(s.n_raw as i32);
            t.n_raw_offset_per.push((region + s.raw_off) as i32);
            t.slot_per.push((region + s.raw_off + s.n_raw) as i32);
            t.n_raw_per_dec.push(s.n_raw_dec as i32);
            t.n_raw_offset_per_dec.push((region + s.raw_off_dec) as i32);
            t.slot_per_dec.push((region + s.raw_off_dec + s.n_raw_dec) as i32);
            for (si, st) in self.stores.iter().enumerate() {
                let r = s.comp[si];
                let ts = &mut t.stores[si];
                let block = st.slot_block(slot) + (s.pos / st.ratio - pos0 / st.ratio);
                ts.n_comp_per.push(r.n_comp as i32);
                ts.comp_base_per.push(r.base as i32);
                ts.keys_base_per.push(r.base);
                ts.state_base_per.push((block * st.ratio * st.width) as i32);
                ts.state_idx_per.push(block as i32);
                if (s.pos + 1) % st.ratio == 0 {
                    if r.n_comp >= r.cap {
                        return Err(eyre!(
                            "kv arena: slot {slot} store L{} region full ({} rows) at pos {}",
                            st.layer, r.cap, s.pos
                        ));
                    }
                    ts.fire_rows.push(b as i32);
                    ts.fire_state_idx.push(block as i32);
                    ts.fire_dst_row.push((r.base + r.n_comp) as i32);
                    ts.fire_comp_pos.push((s.pos + 1 - st.ratio) as i32);
                }
            }
        }
        Ok(t)
    }

    /// Rows per slot in a step's `slots` list (runs of one slot; `tables`
    /// refuses anything else).
    pub fn step_rows(slots: &[u32]) -> Vec<(u32, u32)> {
        let mut v: Vec<(u32, u32)> = Vec::new();
        for &s in slots {
            match v.last_mut() {
                Some((l, n)) if *l == s => *n += 1,
                _ => v.push((s, 1)),
            }
        }
        v
    }

    /// Compact every stream of the step whose raw region cannot take its rows
    /// (`needs_compaction_rows`). Call before `tables`.
    pub fn compact_for_step(&mut self, slots: &[u32], stream: &Stream, scratch: &mut DeviceBuffer<u16>) -> eyre::Result<()> {
        for (slot, rows) in Self::step_rows(slots) {
            if self.needs_compaction_rows(slot, rows) {
                self.compact_raw(slot, stream, scratch)?;
            }
        }
        Ok(())
    }

    /// True when the next append of `slot` would run off its raw region: the
    /// caller must `compact_raw` first (a D2D copy per layer, on `stream`).
    pub fn needs_compaction(&self, slot: u32) -> bool {
        self.needs_compaction_rows(slot, 1)
    }

    /// `needs_compaction` for a step that appends `rows` rows of `slot`.
    pub fn needs_compaction_rows(&self, slot: u32, rows: u32) -> bool {
        self.stream(slot).is_some_and(|s| {
            (s.raw_off + s.n_raw + rows) as usize > ARENA_RAW_ROWS || (s.raw_off_dec + s.n_raw_dec + rows) as usize > ARENA_RAW_ROWS
        })
    }

    /// Move `slot`'s live window to the start of its region in every layer, the
    /// same two-hop copy `forward_layer` and `normalize_raw_windows` do, using
    /// `scratch` (>= `SWA_WINDOW * N_HEAD_DIM` f16) as the bounce buffer.
    pub fn compact_raw(&mut self, slot: u32, stream: &Stream, scratch: &mut DeviceBuffer<u16>) -> eyre::Result<()> {
        let (raw_off, n_raw, raw_off_dec, n_raw_dec) = {
            let s = self.stream(slot).ok_or_else(|| eyre!("kv arena: slot {slot} not live"))?;
            (s.raw_off, s.n_raw, s.raw_off_dec, s.n_raw_dec)
        };
        let hd = N_HEAD_DIM as usize;
        let region = Self::raw_region_base(slot) as usize * hd;
        self.dgpu.set_current()?;
        for (l, ls) in self.state.layers.iter_mut().enumerate() {
            let (off, n) = if l < CED_DECODER_START { (raw_off, n_raw) } else { (raw_off_dec, n_raw_dec) };
            if off == 0 || n == 0 {
                continue;
            }
            let win = n as usize * hd;
            if scratch.len() < win {
                return Err(eyre!("kv arena: compaction scratch {} < window {}", scratch.len(), win));
            }
            let buf = &mut ls.kv_cache;
            {
                let src = buf.slice_view(region + off as usize * hd, win);
                let mut sc = scratch.slice_view_mut(0, win);
                sc.copy_from_buffer_async(&src, stream)?;
            }
            {
                let sc = scratch.slice_view(0, win);
                let mut dst = buf.slice_view_mut(region, win);
                dst.copy_from_buffer_async(&sc, stream)?;
            }
        }
        if let Some(s) = self.stream_mut(slot) { s.raw_off = 0; s.raw_off_dec = 0; }
        Ok(())
    }

    /// Advance `slot` by one appended token: the raw window (monotonic append
    /// with the SWA_WINDOW cap, as `forward_layer` keeps it), each store's
    /// counters where its boundary fired at this position, and `pos`.
    pub fn advance(&mut self, slot: u32) -> eyre::Result<()> {
        let ratios: Vec<u32> = self.ratios().collect();
        let s = self.stream_mut(slot).ok_or_else(|| eyre!("kv arena: slot {slot} not live"))?;
        s.step(ratios);
        Ok(())
    }

    /// After a step: keep the first `keep` of the rows `slot` ran (1 for a
    /// plain decode row; DSpark keeps the accepted prefix). Advances the
    /// counters `keep` positions, which is the whole rollback: nothing past
    /// the counters is ever read, so the rejected rows' raw KV, comp rows and
    /// keys are dead (plan 3.4). Then the commit: a store whose group is still
    /// open at the new position has that group's rows in the block the step
    /// wrote them to, and the next step expects them in the slot's first block
    /// (`blocks_per_slot`); one block copy on `stream`, after the step's
    /// kernels (same stream). A one-row step never copies (its group's block
    /// IS the first). `keep` must not exceed the rows the stream ran.
    pub fn accept(&mut self, slot: u32, keep: u32, stream: &Stream) -> eyre::Result<()> {
        if keep == 0 || keep > ARENA_ROWS_PER_STREAM {
            return Err(eyre!("kv arena: accept of {keep} rows for slot {slot}"));
        }
        let pos0 = self.stream(slot).ok_or_else(|| eyre!("kv arena: slot {slot} not live"))?.pos;
        for _ in 0..keep {
            self.advance(slot)?;
        }
        let copies = self.commit_copies(slot, pos0, keep);
        if !copies.is_empty() {
            self.dgpu.set_current()?;
        }
        for (from, to, layer, block) in copies {
            let cs = self.state.layers[layer].compressor.as_mut().ok_or_else(|| {
                eyre!("kv arena: store L{layer} is lent out (accept between steps)")
            })?;
            for buf in [&mut cs.state_kv, &mut cs.state_score] {
                let src = buf.slice_view(from, block);
                let mut dst = buf.slice_view_mut(to, block);
                dst.copy_from_buffer_async(&src, stream)?;
            }
        }
        Ok(())
    }

    /// The commit's block copies for a stream that ran from `pos0` and kept
    /// `keep` rows: `(from, to, layer, len)` in floats of the store's
    /// accumulator buffers. Pure, for `accept` and the unit test.
    fn commit_copies(&self, slot: u32, pos0: u32, keep: u32) -> Vec<(usize, usize, usize, usize)> {
        let pos1 = pos0 + keep;
        self.stores
            .iter()
            .filter_map(|st| {
                let rel = (pos1 - 1) / st.ratio - pos0 / st.ratio;
                if pos1 % st.ratio == 0 || rel == 0 {
                    return None;
                }
                let block = (st.ratio * st.width) as usize;
                Some(((st.slot_block(slot) + rel) as usize * block, st.slot_block(slot) as usize * block, st.layer, block))
            })
            .collect()
    }

    /// Restore `slot` to a saved copy of its counters (speculative rollback;
    /// the accumulators are the caller's, per `KvMark::per_layer_comp`).
    pub fn restore(&mut self, slot: u32, saved: &StreamKv) -> eyre::Result<()> {
        let s = self.stream_mut(slot).ok_or_else(|| eyre!("kv arena: slot {slot} not live"))?;
        if s.comp.iter().zip(&saved.comp).any(|(a, b)| a.base != b.base || a.cap != b.cap) {
            return Err(eyre!("kv arena: restore of slot {slot} with different regions"));
        }
        *s = saved.clone();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_list_first_fit_and_coalesce() {
        let mut fl = RowFreeList::new(100);
        assert_eq!(fl.carve(30), Some(0));
        assert_eq!(fl.carve(30), Some(30));
        assert_eq!(fl.carve(50), None);
        fl.give_back(0, 30);
        assert_eq!(fl.free, vec![(0, 30), (60, 40)]);
        assert_eq!(fl.carve(40), Some(60));
        fl.give_back(30, 30);
        assert_eq!(fl.free, vec![(0, 60)]);
        assert_eq!(fl.free_rows(), 60);
        assert!(fl.fits(60) && !fl.fits(61));
    }

    #[test]
    fn free_runs_is_the_complement() {
        assert_eq!(free_runs(vec![], 10), vec![(0, 10)]);
        assert_eq!(free_runs(vec![(3, 2), (0, 1)], 10), vec![(1, 2), (5, 5)]);
        assert_eq!(free_runs(vec![(0, 4), (4, 6)], 10), vec![]);
        assert_eq!(free_runs(vec![(2, 5), (4, 2)], 9), vec![(0, 2), (7, 2)], "overlap tolerated");
    }

    #[test]
    fn carve_at_only_from_the_start_of_a_run() {
        let mut fl = RowFreeList::new(100);
        assert_eq!(fl.carve(30), Some(0));
        assert!(!fl.carve_at(20, 5), "inside a live region");
        assert!(!fl.carve_at(31, 5), "not the start of the run");
        assert!(!fl.carve_at(30, 71), "longer than the run");
        assert!(fl.carve_at(30, 10));
        assert_eq!(fl.free, vec![(40, 60)]);
        assert!(fl.carve_at(40, 60));
        assert!(fl.free.is_empty());
    }

    /// Apply `move_chunks` through a bounce buffer on a host "store", exactly
    /// as `move_region_rows` does on the device (src -> bounce -> dst per chunk).
    fn apply_move(buf: &mut [i64], from: u32, to: u32, n: u32, chunk: u32) {
        for (s, d, len) in move_chunks(from, to, n, chunk) {
            let bounce: Vec<i64> = buf[s as usize..(s + len) as usize].to_vec();
            buf[d as usize..(d + len) as usize].copy_from_slice(&bounce);
        }
    }

    #[test]
    fn move_chunks_overlapping_both_directions() {
        for &(from, to, n, chunk) in &[(10u32, 3u32, 20u32, 4u32), (3, 10, 20, 4), (0, 1, 50, 7), (1, 0, 50, 7), (5, 45, 30, 8), (5, 5, 9, 2), (0, 7, 7, 3)] {
            let mut buf: Vec<i64> = (0..100).map(|i| -(i as i64) - 1).collect();
            for r in 0..n {
                buf[(from + r) as usize] = 1000 + r as i64;
            }
            apply_move(&mut buf, from, to, n, chunk);
            for r in 0..n {
                assert_eq!(buf[(to + r) as usize], 1000 + r as i64, "from {from} to {to} n {n} chunk {chunk} row {r}");
            }
        }
    }

    /// Tiny LCG so the property test needs no dev-dependency.
    fn lcg(s: &mut u64) -> u32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 33) as u32
    }

    #[test]
    fn compact_around_keeps_every_row_and_frees_after_x() {
        let mut seed = 7u64;
        for case in 0..2000 {
            let rows_cap = 40 + lcg(&mut seed) % 200;
            // Random live regions carved first-fit with holes punched between.
            let mut fl = RowFreeList::new(rows_cap);
            let mut regs: Vec<(u32, u32, u32)> = Vec::new(); // (base, cap, n)
            for _ in 0..(1 + lcg(&mut seed) % 8) {
                let cap = 1 + lcg(&mut seed) % 30;
                if let Some(b) = fl.carve(cap) {
                    regs.push((b, cap, lcg(&mut seed) % (cap + 1)));
                }
            }
            if regs.is_empty() {
                continue;
            }
            let mut i = 0;
            while i < regs.len() && regs.len() > 1 {
                if lcg(&mut seed) % 3 == 0 {
                    let (b, c, _) = regs.remove(i);
                    fl.give_back(b, c);
                } else {
                    i += 1;
                }
            }
            let mut buf = vec![-1i64; rows_cap as usize];
            for (k, &(b, _, n)) in regs.iter().enumerate() {
                for r in 0..n {
                    buf[(b + r) as usize] = (k as i64) * 10_000 + r as i64;
                }
            }
            let x = (lcg(&mut seed) as usize) % regs.len();
            let plan_in: Vec<(u32, u32)> = regs.iter().map(|&(b, c, _)| (b, c)).collect();
            let (moves, free) = plan_compact_around(&plan_in, x, rows_cap);
            let chunk = 1 + lcg(&mut seed) % 6;
            let mut now: Vec<u32> = regs.iter().map(|r| r.0).collect();
            for (i, nb) in &moves {
                apply_move(&mut buf, now[*i], *nb, regs[*i].2, chunk);
                now[*i] = *nb;
            }
            assert_eq!(moves.len(), regs.len(), "case {case}: every region placed once");
            for (k, &(_, c, n)) in regs.iter().enumerate() {
                assert!(now[k] + c <= rows_cap, "case {case}");
                for r in 0..n {
                    assert_eq!(buf[(now[k] + r) as usize], (k as i64) * 10_000 + r as i64, "case {case} region {k} row {r}");
                }
            }
            // No two regions overlap, and the free run is every free row, right after x.
            let mut spans: Vec<(u32, u32)> = regs.iter().enumerate().map(|(k, &(_, c, _))| (now[k], now[k] + c)).collect();
            spans.sort();
            for w in spans.windows(2) {
                assert!(w[0].1 <= w[1].0, "case {case}: overlap {w:?}");
            }
            assert_eq!(free.0, now[x] + regs[x].1, "case {case}: free run starts after x");
            assert_eq!(free.1, fl.free_rows(), "case {case}: free run holds every free row");
            assert!(spans.iter().all(|&(a, b)| b <= free.0 || a >= free.0 + free.1), "case {case}: free run overlaps a region");
        }
    }

    /// A `KvArena` with stores and streams but no device buffers: `tables`,
    /// `can_step_rows`, `advance` and `commit_copies` touch only host state.
    #[cfg(feature = "v41")]
    fn host_arena(n_slots: u32, rows_cap: u32) -> KvArena {
        let stores = KV_SOURCE_LAYERS
            .iter()
            .map(|&l| {
                let ratio = COMPRESS_RATIOS[l as usize];
                CompStore { layer: l as usize, ratio, width: N_HEAD_DIM, rows_cap, free: RowFreeList::new(rows_cap), blocks_per_slot: blocks_per_slot(ratio) }
            })
            .collect();
        KvArena {
            dgpu: Device::new(0),
            n_slots,
            state: HetModelState { layers: Vec::new(), n_kv_max: 0 },
            stores,
            streams: vec![None; n_slots as usize],
        }
    }

    /// DSPARK_ARENA_PLAN 3.1-3.4 on the host: a stream runs steps of 1..=8
    /// rows (positions pos..pos+R) and keeps a random prefix. A model of the
    /// raw region and of the accumulator blocks replays exactly what the
    /// kernels do with the tables (append at `slot_per`; attend the driver's
    /// window; state-write `state_base_per + (q % ratio) * width`; pool block
    /// `fire_state_idx`; then `accept`'s block copies), keeping the rejected
    /// rows' writes in place. Every row must see exactly its causal window
    /// (positions q-W+1..=q, in order) and every fire must pool exactly its
    /// group, into the next comp row of the region. A second stream steps one
    /// row at a time in the same steps (rows before and after the run) and
    /// must get the tables it gets alone.
    #[cfg(feature = "v41")]
    #[test]
    fn multi_row_tables_match_one_row_steps() {
        let mut ar = host_arena(3, 100_000);
        let pos_a = 37u32; // odd: the first step carries an open ratio-2 group
        let pos_b = 200u32;
        for (slot, pos) in [(0u32, pos_a), (1, pos_b)] {
            let comp = ar.stores.iter().map(|st| CompRegion {
                base: 100 + slot * 30_000,
                cap: 30_000,
                n_comp: pos / st.ratio,
                n_index_comp: pos / st.ratio,
            }).collect();
            let n_raw = pos.min(SWA_WINDOW);
            ar.streams[slot as usize] = Some(StreamKv { pos, raw_off: 0, n_raw, raw_off_dec: 0, n_raw_dec: n_raw, comp });
        }
        let region = |slot: u32| KvArena::raw_region_base(slot) as usize;
        let rows = ARENA_RAW_ROWS * 3;
        // raw[row] = the position whose KV the row holds.
        let mut raw: Vec<Option<u32>> = vec![None; rows];
        for (slot, pos) in [(0u32, pos_a), (1, pos_b)] {
            let s = ar.stream(slot).unwrap();
            for i in 0..s.n_raw {
                raw[region(slot) + i as usize] = Some(pos - s.n_raw + i);
            }
        }
        // blocks[store][block][row] = the position the accumulator row holds.
        let mut blocks: Vec<Vec<Vec<Option<u32>>>> = ar.stores.iter()
            .map(|st| vec![vec![None; st.ratio as usize]; (3 * st.blocks_per_slot) as usize]).collect();
        // The carried open group of stream 0 (pos 37 odd: position 36 in row 0).
        for (si, st) in ar.stores.iter().enumerate() {
            for (slot, pos) in [(0u32, pos_a), (1, pos_b)] {
                let g = pos - pos % st.ratio;
                for q in g..pos {
                    blocks[si][st.slot_block(slot) as usize][(q % st.ratio) as usize] = Some(q);
                }
            }
        }
        let mut rng = 0x5eed_u64;
        let mut lcg = |m: u32| -> u32 { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((rng >> 33) as u32) % m };
        for step in 0..3000 {
            let r = 1 + lcg(ARENA_ROWS_PER_STREAM);
            let keep = 1 + lcg(r);
            let b_first = lcg(2) == 0;
            let mut slots: Vec<u32> = vec![0; r as usize];
            if b_first { slots.insert(0, 1) } else { slots.push(1) }
            assert!(ar.can_step_rows(0, r));
            for (slot, n) in KvArena::step_rows(&slots) {
                if ar.needs_compaction_rows(slot, n) {
                    // compact_raw on the model: both windows to the region start.
                    let s = ar.stream_mut(slot).unwrap();
                    assert_eq!((s.raw_off, s.n_raw), (s.raw_off_dec, s.n_raw_dec));
                    let (off, n_raw) = (s.raw_off as usize, s.n_raw as usize);
                    let base = KvArena::raw_region_base(slot) as usize;
                    let win: Vec<_> = raw[base + off..base + off + n_raw].to_vec();
                    raw[base..base + n_raw].copy_from_slice(&win);
                    s.raw_off = 0;
                    s.raw_off_dec = 0;
                }
            }
            let t = ar.tables(&slots).unwrap();
            // Stream 1's row equals its one-row tables.
            let alone = ar.tables(&[1]).unwrap();
            let b1 = if b_first { 0 } else { r as usize };
            assert_eq!(t.pos_per[b1], alone.pos_per[0]);
            assert_eq!(t.slot_per[b1], alone.slot_per[0]);
            assert_eq!(t.n_raw_per[b1], alone.n_raw_per[0]);
            for (ts, ta) in t.stores.iter().zip(&alone.stores) {
                assert_eq!(ts.state_base_per[b1], ta.state_base_per[0]);
                assert_eq!(ts.n_comp_per[b1], ta.n_comp_per[0]);
            }
            let pos0 = ar.stream(0).unwrap().pos;
            // The kernels: raw append for every row, then each row's window.
            for b in 0..slots.len() {
                raw[t.slot_per[b] as usize] = Some(t.pos_per[b] as u32);
                assert_eq!(t.slot_per[b], t.slot_per_dec[b]);
            }
            for b in 0..slots.len() {
                let q = t.pos_per[b] as u32;
                if slots[b] == 0 {
                    assert_eq!(q, pos0 + (b - usize::from(b_first)) as u32, "step {step}: row positions");
                }
                let n = (t.n_raw_per[b] as u32 + 1).min(SWA_WINDOW);
                let off = (t.slot_per[b] as u32 + 1 - n) as usize;
                let want: Vec<Option<u32>> = (q + 1 - n..=q).map(Some).collect();
                assert_eq!(&raw[off..off + n as usize], &want[..], "step {step} row {b}: raw window");
                assert!(off >= region(slots[b]) && off + (n as usize) <= region(slots[b]) + ARENA_RAW_ROWS);
            }
            // The compressor: state-write every row, then pool every fire.
            for (si, st) in ar.stores.iter().enumerate() {
                let ts = &t.stores[si];
                for b in 0..slots.len() {
                    let q = t.pos_per[b] as u32;
                    let blk = ts.state_base_per[b] as u32 / (st.ratio * st.width);
                    assert_eq!(blk as i32, ts.state_idx_per[b]);
                    let lo = st.slot_block(slots[b]);
                    assert!(blk >= lo && blk < lo + st.blocks_per_slot, "step {step}: block in the slot's range");
                    blocks[si][blk as usize][(q % st.ratio) as usize] = Some(q);
                }
                for (k, &fr) in ts.fire_rows.iter().enumerate() {
                    let q = t.pos_per[fr as usize] as u32;
                    let got = &blocks[si][ts.fire_state_idx[k] as usize];
                    let want: Vec<Option<u32>> = (q + 1 - st.ratio..=q).map(Some).collect();
                    assert_eq!(got, &want, "step {step} L{}: pooled group of pos {q}", st.layer);
                    assert_eq!(ts.fire_comp_pos[k] as u32, q + 1 - st.ratio);
                    let base = 100 + slots[fr as usize] * 30_000;
                    assert_eq!(ts.fire_dst_row[k] as u32, base + (q + 1) / st.ratio - 1, "step {step}: comp row of pos {q}");
                    assert_eq!(ts.n_comp_per[fr as usize] as u32 + 1, (q + 1) / st.ratio);
                }
            }
            // Accept: keep a prefix of stream 0, the one row of stream 1.
            for (slot, k) in [(0u32, keep), (1, 1)] {
                let p = ar.stream(slot).unwrap().pos;
                for (from, to, layer, len) in ar.commit_copies(slot, p, k) {
                    let si = ar.stores.iter().position(|st| st.layer == layer).unwrap();
                    let bl = (ar.stores[si].ratio * ar.stores[si].width) as usize;
                    assert_eq!(len, bl);
                    blocks[si][to / bl] = blocks[si][from / bl].clone();
                }
                for _ in 0..k {
                    ar.advance(slot).unwrap();
                }
                assert_eq!(ar.stream(slot).unwrap().pos, p + k);
                if slot == 1 {
                    assert!(ar.commit_copies(1, p, 1).is_empty(), "a one-row step never copies");
                }
            }
        }
        // Refusals.
        assert!(ar.tables(&[0, 1, 0]).is_err(), "a slot split by another slot's rows");
        assert!(ar.tables(&[0; ARENA_ROWS_PER_STREAM as usize + 1]).is_err(), "more rows than the blocks hold");
        assert!(!ar.can_step_rows(0, ARENA_ROWS_PER_STREAM + 1));
    }
}
